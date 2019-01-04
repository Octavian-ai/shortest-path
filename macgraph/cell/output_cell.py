
import tensorflow as tf

from ..minception import *
from ..args import ACTIVATION_FNS
from ..util import *
from ..layers import *
from ..attention import *
from ..component import *


class OutputCell(Component):

	def __init__(self, args):
		super().__init__(args, "output_cell")

		# TODO: generate this from components
		tr = []
		for i in range(args["mp_read_heads"]):
			tr.append(f"mp{i}")

		for i in range(args["max_decode_iterations"]):
			tr.append(f"po{i}")


		self.output_table = Tensor("table")
		self.output_query = Tensor("focus_query")
		self.focus = AttentionByIndex(args, 
			self.output_table, self.output_query, seq_len=6, 
			table_representation=tr,
			name="focus")

		

	def forward(self, features, context, mp_reads):

		with tf.name_scope(self.name):

			in_all = []

			def add(t):
				in_all.append(pad_to_len_1d(t, self.args["embed_width"]))

			def add_all(t):
				for i in t:
					add(i)

			add_all(mp_reads)

			prev_outputs = tf.unstack(context.in_prev_outputs, axis=1)
			add_all(prev_outputs)
			
			in_stack = tf.stack(in_all, axis=1)
			in_stack = dynamic_assert_shape(in_stack, [features["d_batch_size"], len(in_all), self.args["embed_width"]])

			self.output_table.bind(in_stack)
			self.output_query.bind(context.in_iter_id)
			v = self.focus.forward(features)
			v.set_shape([None, self.args["embed_width"]])

			for i in range(self.args["output_layers"]):
				v = layer_dense(v, self.args["output_width"], self.args["output_activation"], name=f"output{i}")

			return v



