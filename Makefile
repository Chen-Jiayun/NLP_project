step-1:
	python3 step-1.py > result/stpe-1-raw.txt
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

step-3-ngram:
	python3 step-3-n-gram.py > result/step-3-n-gram.json
	@echo "==== step-3-ngram:done ===="

step-3-skipgram:
	python3 step-3-skip-gram.py > result/step-3-skip-gram.json
	@echo "==== step-3-skipgram:done ===="

step-4-tSNE:
	python3 step-4_1-tSNE.py

step-4-biotSNE:
	python3 step-4_2-biotSNE.py

step-4-cocurrence:
	python3 step-4_3-cocurrence.py

step-4-similar:
	python3 step-4_4-similar.py