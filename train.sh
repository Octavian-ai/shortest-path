#!/bin/bash

task=StationShortestCount
iteration=10

python -m macgraph.train \
	--dataset $task \
	--tag iter_$iteration \
	--tag upto_9 \
	--tag r$RANDOM \
	--filter-output-class 0 \
	--filter-output-class 1 \
	--filter-output-class 2 \
	--filter-output-class 3 \
	--filter-output-class 4 \
	--filter-output-class 5 \
	--filter-output-class 6 \
	--filter-output-class 7 \
	--filter-output-class 8 \
	--filter-output-class 9 \
	--train-max-steps 10 \
	--max-decode-iterations $iteration \
	--fast \
	$@