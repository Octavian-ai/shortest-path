# Finding shortest paths with Graph Networks

> In this article we show how a Graph Network with attention read and write can perform shortest path calculations. This network performs this task with 99.91% accuracy after minimal training.

Here at Octavian we believe that graphs are a powerful medium for representing diverse knowledge (for example BenevolentAI uses them to represent pharmaceutical research and knowledge).

Neural networks are a way to create functions that no human could write. They do this by harnessing the power of large datasets. 

On problems for which we have capable neural models, we can use example inputs and outputs to train the network to learn a function that transforms those inputs into those outputs, and hopefully generalizes to other unseen inputs.

We need to be able to build neural networks that can learn functions on graphs. Those neural networks need the right inductive biases so that they can reliably learn useful graph functions. With that foundation, we can build powerful neural graph systems.

Here we present a "Graph network with attention read and write", a simple network that can effectively compute shortest path. It is an example of how to combine different neural network components to make a system that readily learns a classical graph algorithm.

We present this network both as a novel system in of itself, but more importantly as the basis for further investigation into effective neural graph computation.

See the [full descripton on Medium]()

Download the [pre-compiled YAML dataset](https://storage.googleapis.com/octavian-static/download/clevr-graph/StationShortestCount.zip) or the [fully-compiled TFRecords dataset](https://storage.googleapis.com/octavian-static/download/mac-graph/StationShortestCount.zip). Data is expected to live in input_data/processed/StationShortestCount/.

## Running

`shell
pipenv install
pipenv run ./train.sh
`

## Visualising the attention

`shell
pipenv run python -m macgraph.predict --model-dir ./output/StationShortestPath/<insert your trained model path>
`
