
import tensorflow as tf

from .mac_cell import *
from ..util import *




def static_decode(args, features, inputs, labels, vocab_embedding):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):

		d_cell = MAC_RNNCell(args, features, vocab_embedding)
		d_cell_initial = d_cell.zero_state(dtype=tf.float32, batch_size=features["d_batch_size"])
		d_cell_empty_output = [tf.zeros([features["d_batch_size"], args["output_width"]])]

		# Hard-coded unroll of the reasoning network for simplicity
		states = [(d_cell_empty_output, d_cell_initial)]
		for i in range(args["max_decode_iterations"]):
			with tf.variable_scope("decoder_cell", reuse=tf.AUTO_REUSE):
				inputs_slice = [item[i] for item in inputs]
				prev_outputs = [item[0][0] for item in states]

				if len(prev_outputs) < args["max_decode_iterations"]:
					for i in range(args["max_decode_iterations"] - len(prev_outputs)):
						prev_outputs.append(d_cell_empty_output[0])

				assert len(prev_outputs) == args["max_decode_iterations"]
				prev_outputs = tf.stack(prev_outputs, axis=1)
				
				inputs_for_iteration = [*inputs_slice, prev_outputs]
				prev_state = states[-1][1]

				states.append(d_cell(inputs_for_iteration, prev_state))

		final_output = states[-1][0][0]

		def get_tap(idx, key):
			with tf.name_scope(f"get_tap_{key}"):
				tap = [i[0][idx] for i in states[1:] if i[0] is not None]

				for i in tap:
					if i is None:
						return None

				if len(tap) == 0:
					return None

				tap = tf.convert_to_tensor(tap)

				 # Deal with batch vs iteration axis layout
				if len(tap.shape) == 3:
					tap = tf.transpose(tap, [1,0,2]) # => batch, iteration, data
				if len(tap.shape) == 4:
					tap = tf.transpose(tap, [1,0,2,3]) # => batch, iteration, control_head, data
					
				return tap

		out_taps = {
			key: get_tap(idx+1, key)
			for idx, key in enumerate(d_cell.tap_sizes().keys())
		}
		
		return final_output, out_taps


def execute_reasoning(args, features, **kwargs):

	d_eye = tf.eye(args["max_decode_iterations"])

	iteration_id = [
		tf.tile(tf.expand_dims(d_eye[i], 0), [features["d_batch_size"], 1])
		for i in range(args["max_decode_iterations"])
	]

	inputs = [iteration_id]

	final_output, out_taps = static_decode(args, features, inputs, **kwargs)


	final_output = dynamic_assert_shape(final_output, [features["d_batch_size"], args["output_width"]])


	return final_output, out_taps




