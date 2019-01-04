
import tensorflow as tf

from ..component import Component

from .output_cell import *
from .messaging_cell import *
from .types import *

from ..util import *
from ..minception import *
from ..layers import *

class MAC_RNNCell(tf.nn.rnn_cell.RNNCell):

	def __init__(self, args, features,  vocab_embedding):

		self.args = args
		self.features = features
		self.vocab_embedding = vocab_embedding

		self.mac = MAC_Component(args)

		super().__init__(self)


	def __call__(self, inputs, in_state):
		'''Build this cell (part of implementing RNNCell)

		This is a wrapper that marshalls our named taps, to 
		make sure they end up where we expect and are present.
		
		Args:
			inputs: `2-D` tensor with shape `[batch_size, input_size]`.
			state: if `self.state_size` is an integer, this should be a `2-D Tensor`
				with shape `[batch_size, self.state_size]`.	Otherwise, if
				`self.state_size` is a tuple of integers, this should be a tuple
				with shapes `[batch_size, s] for s in self.state_size`.
			scope: VariableScope for the created subgraph; defaults to class name.
		Returns:
			A pair containing:
			- Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
			- New state: Either a single `2-D` tensor, or a tuple of tensors matching
				the arity and shapes of `state`.
		'''

		output, out_state = self.mac.forward(self.features, inputs, in_state, self.vocab_embedding)

		taps = self.mac.all_taps()

		out_data = [output]

		for k,v in taps.items():
			out_data.append(v)

		return out_data, out_state


	def tap_sizes(self):
		return self.mac.all_tap_sizes()



	@property
	def state_size(self):
		"""
		Returns a size tuple
		"""
		return (
			tf.TensorShape([self.args["kb_node_max_len"], self.args["mp_state_width"]]),
		)

	@property
	def output_size(self):

		tap_sizes = self.mac.all_tap_sizes()

		return [
			self.args["output_width"], 
		] + tap_sizes




class MAC_Component(Component):

	def __init__(self, args):
		super().__init__(args, name=None) # empty to preserve legacy naming

		self.output_cell = OutputCell(args)

	"""
	Special forward. Should return output, out_state
	"""
	def forward(self, features, inputs, in_state, vocab_embedding):
		# TODO: remove this transition scaffolding
		self.features = features
		self.vocab_embedding = vocab_embedding

		with tf.variable_scope("mac_cell", reuse=tf.AUTO_REUSE):

			in_node_state = in_state[0]

			in_iter_id = inputs[0]
			in_iter_id = dynamic_assert_shape(in_iter_id, [self.features["d_batch_size"], self.args["max_decode_iterations"]], "in_iter_id")

			in_prev_outputs = inputs[-1]

			embedded_question  = tf.nn.embedding_lookup(vocab_embedding, features["src"])
			embedded_question *= tf.sqrt(tf.cast(self.args["embed_width"], embedded_question.dtype)) # As per Transformer model
			embedded_question = dynamic_assert_shape(embedded_question, [features["d_batch_size"], features["src_len"], self.args["embed_width"]])

			context = CellContext(
				features=self.features, 
				args=self.args,
				vocab_embedding=self.vocab_embedding,
				in_prev_outputs=in_prev_outputs,
				in_iter_id=in_iter_id,
				in_node_state=in_node_state,
				embedded_question=embedded_question
			)
		
			mp_reads, out_mp_state, mp_taps = messaging_cell(context)
			
			output = self.output_cell.forward(features, context, mp_reads)	
	
			# TODO: tidy away later
			self.mp_taps = mp_taps
			
			self.mp_state = out_mp_state
			self.context = context
			
			out_state = (out_mp_state,)


			return output, out_state


	def taps(self):

		# TODO: Remove all of this and let it run in the subsystem

		mp_taps = self.mp_taps
		
		empty_attn = tf.fill([self.features["d_batch_size"], self.args["max_seq_len"], 1], 0.0)
		empty_query = tf.fill([self.features["d_batch_size"], self.args["max_seq_len"]], 0.0)


		# TODO: AST this all away
		out_taps = {
			"mp_node_state":			self.mp_state,
			"iter_id":					self.context.in_iter_id,
		}

		mp_reads = [f"mp_read{i}" for i in range(self.args["mp_read_heads"])]

		suffixes = ["_attn", "_attn_raw", "_query", "_signal"]
		for qt in ["token_index_attn"]:
			suffixes.append("_query_"+qt)

		for mp_head in ["mp_write", *mp_reads]:
			for suffix in suffixes:
				i = mp_head + suffix
				out_taps[i] = mp_taps.get(i, empty_query)

	

		return out_taps




	def tap_sizes(self):

		t = {
			"mp_node_state":			tf.TensorShape([self.args["kb_node_max_len"], self.args["mp_state_width"]]),
			"iter_id":					self.args["max_decode_iterations"],
		}

		mp_reads = [f"mp_read{i}" for i in range(self.args["mp_read_heads"])]

		for mp_head in ["mp_write", *mp_reads]:
			t[f"{mp_head}_attn"]			= self.args["kb_node_max_len"]
			t[f"{mp_head}_attn_raw"] 		= self.args["kb_node_max_len"]
			t[f"{mp_head}_query"]			= self.args["kb_node_width"] * self.args["embed_width"]
			t[f"{mp_head}_signal"]			= self.args["mp_state_width"]
			t[f"{mp_head}_query_token_index_attn"  ] = self.args["max_seq_len"]

		return t







