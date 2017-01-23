for file in tsa_datasets/*; do
	echo $file
	python emotion_analysis.py -d $file | tee "emotion_results/$(basename $file).txt"
done
