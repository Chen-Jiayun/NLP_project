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

play-toy:
	python3 toy-play.py
