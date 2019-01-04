

from typing import *

# Someday TensorFlow will have types!
Tensor = Any

class CellContext(NamedTuple):
	features: Dict
	args: Dict
	vocab_embedding: Tensor
	in_prev_outputs: Tensor
	in_iter_id: Tensor
	in_node_state: Tensor
	embedded_question: Tensor