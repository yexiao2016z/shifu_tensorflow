package ml.shifu.shifu.keras;
//import org.tensorflow.*;
//import org.deeplearning4j
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.Enumeration;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import ml.shifu.guagua.hadoop.util.HDPUtils;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Environment;
import parquet.Log;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.apache.pig.ExecType;
import org.apache.pig.PigServer;
import org.omg.SendingContext.RunTime;
//import org.apache.spark.api.java.JavaRDD;
//import org.apache.spark.api.java.JavaSparkContext;
//import org.datavec.api.util.ClassPathResource;
//import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
//import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
//import org.deeplearning4j.spark.api.*;
//import org.deeplearning4j.spark.impl.multilayer.*;
//import org.deeplearning4j.spark.impl.paramavg.*;

public class KerasExecutor {
	private static Logger log = LoggerFactory.getLogger(KerasExecutor.class);
    private static KerasExecutor instance = new KerasExecutor();

    // avoid to create instance, used as singleton
    private KerasExecutor() {
    }

    /**
     * Get the pig executor handler
     * 
     * @return - executor handler
     */
    public static KerasExecutor getExecutor() {
        return instance;
    }
    
    public void submitJob(ModelConfig modelConfig, String kerasScriptPath, SourceType sourceType, PathFinder pathFinder, String modelFilePath, String select_status)
    		throws IOException, InterruptedException {
    	//if(sourceType.equals(SourceType.LOCAL)) {
	    	String loss = ((String)modelConfig.getParams().get("Loss")).toLowerCase();
	    	String optimizer = ((String)modelConfig.getParams().get("Optimizer")).toLowerCase();
			String dataPath = pathFinder.getNormalizedDataPath();
			String epochNum = String.valueOf(modelConfig.getNumTrainEpochs());
			String validRate = modelConfig.getValidSetRate().toString();
			String learningRate = modelConfig.getParams().get("LearningRate").toString();
			String baggingNums = new Integer(modelConfig.getBaggingNum()).toString();
			String baggingRate = new Double(modelConfig.getBaggingSampleRate()).toString();
			String baggingReplace = new Boolean(modelConfig.isBaggingWithReplacement()).toString();
			log.info("Loading Parameters Successfully!");
			
			String shifuHome = System.getenv("SHIFU_HOME");
			//StringBuilder cmd = new StringBuilder();
			Process pr;
			//cmd.append(shifuHome + "/TFDependancy/glibc2.17/lib/ld-linux-x86-64.so.2 " + 
			//		"--library-path" + shifuHome + "/TFDependancy/glibc2.17/lib:$LD_LIBRARY_PATH:/lib64 " +
			//		shifuHome + "/TFDependancy/python2.7/bin/python");
			ProcessBuilder pb = new ProcessBuilder(
						shifuHome + "/lib/glibc2.17/lib/ld-linux-x86-64.so.2", 
						"--library-path",shifuHome + "/lib/glibc2.17/lib:$LD_LIBRARY_PATH:/lib64",shifuHome+"/lib/python2.7/bin/python", "-W", "ignore",
						//"python",
					//shifuHome + "/TFDependancy/glibc2.17/lib/ld-linux-x86-64.so.2 " + 
					//"--library-path" + shifuHome + "/TFDependancy/glibc2.17/lib:$LD_LIBRARY_PATH:/lib64 " +
					//shifuHome + "/TFDependancy/python2.7/bin/python",
						kerasScriptPath, modelFilePath, loss, optimizer, dataPath, 
	    				select_status, epochNum, validRate, learningRate, baggingNums, baggingReplace, baggingRate);
			//pb.environment().put("PYTHONPATH", "./site-packages");
			//pb.environment().put("PYTHONHOME", "./python2.7/bin/python");
			log.info("Building Process To Run DNN");
			pr = pb.start();
			InputStream in = pr.getInputStream();
			BufferedReader reader = new BufferedReader(new InputStreamReader(in));
	        String line;
	        while ((line = reader.readLine()) != null) {
	        	log.info(line);
	        }  
	        InputStream e = pr.getErrorStream();
			BufferedReader readere = new BufferedReader(new InputStreamReader(e));
	        String linee;
	        StringBuilder sb = new StringBuilder();
	        while ((linee = readere.readLine()) != null) {
	        	sb.append(linee);
	        	//log.debug(linee)
	        }
	        pr.waitFor();
	        if(pr.exitValue() != 0)
	        	throw new RuntimeException(sb.toString());
		//}catch(Exception e) {
		//	log.info("Error When Excute Script :" + e.getMessage());
		//}
	//}
    }
  
}
