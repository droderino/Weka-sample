import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class Main {

	public static void main(String[] args)
	{
		System.out.println("Weka Sample");
		
		//Import Data
		try {
			//u45onDiabetesData();
			
			DataSource trainSource = new DataSource("/home/user/Frameworks/weka/data/ReutersCorn-train.arff");
			DataSource testSource = new DataSource("/home/user/Frameworks/weka/data/ReutersCorn-test.arff");
			
			Instances train = trainSource.getDataSet();
			Instances test = testSource.getDataSet();
			
			if( train.classIndex() == -1 )
				train.setClassIndex(train.numAttributes()-1);
			if( test.classIndex() == -1 )
				test.setClassIndex(test.numAttributes()-1);
			
			System.out.println("Train NaiveBayes");
			Filter filter = new StringToWordVector();
			
			FilteredClassifier cls = new FilteredClassifier();
			cls.setFilter(filter);
			cls.setClassifier(new NaiveBayes());
			
			cls.buildClassifier(train);
			
			System.out.println("Evaluate with test");
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cls, test);
			System.out.println(eval.toSummaryString());
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	private static void u45onDiabetesData() throws Exception {
		DataSource source = new DataSource("/home/user/Frameworks/weka/data/diabetes.arff");
		Instances data = source.getDataSet();
		
		if(data.classIndex() == -1)
			data.setClassIndex(data.numAttributes()-1);
		
		System.out.println("Train C4.5 tree algorithm");
		String[] options = new String[1];
		options[0] = "-U";
		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(data);
		
		System.out.println("Cross-Validation:");
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(tree, data, 10, new Random(1));
		
		System.out.println( eval.toSummaryString() );
	}
}
