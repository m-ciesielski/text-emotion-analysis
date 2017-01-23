for file in tsa_datasets/*; do
	echo $file
	python sentiment_analysis.py -d $file -p False | tee "results/$(basename $file).txt"
done
