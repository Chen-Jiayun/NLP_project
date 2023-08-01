step-1:
	python3 step-1.py > titles.txt
	@echo "==== step-1:done ===="

step-2-split:
	python3 step-2-split.py > result/step-2-result-split.txt
	@echo "==== step-2-split:done ===="

step-2-nltk:
	python3 step-2-nltk.py > result/step-2-result-nltk.txt
	@echo "==== step-2-nltk:done ===="

step-2-BPE:
	python3 step-2-BPE.py > result/step-2-result-BPE.txt
	@echo "==== step-2-BPE:done ===="

init-toy:
	python3 toy-model.py
	@echo "==== init-toy:done ===="

play-toy:
	python3 toy-play.py

step-3-bigram:
	python3 step-3-bigram.py
	@echo "==== build-model:done ===="

play-bigram-model:
	python3 model-play.py

step-3-ngram:
	python3 step-3-n-gram-main.py > result/step-3-n-gram.json