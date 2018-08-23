package ml.shifu.yarn;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.util.Time;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.client.api.YarnClientApplication;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.codehaus.jackson.map.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.type.TypeReference;

import ml.shifu.shifu.util.JSONUtils;


public class Client {
	//char[]  
	private static final Logger LOG = LoggerFactory.getLogger(Client.class);
	final long MAX_WAIT_TIME = 60000*100;
	public void run(String dataPathDir, String shifuVersion) {
		FileSystem fs = null;
		String appName = "test";
		try {
			;
			YarnClient yarnClient = YarnClient.createYarnClient();
			Configuration conf = new Configuration();
			yarnClient.init(conf);
			yarnClient.start();
			YarnClientApplication app = yarnClient.createApplication();
			
			ApplicationSubmissionContext appContext = app.getApplicationSubmissionContext();
			ApplicationId appId = appContext.getApplicationId();

			appContext.setKeepContainersAcrossApplicationAttempts(false);
			appContext.setApplicationName(appName);

			
			Map<String, LocalResource> localResources = new HashMap<String, LocalResource>();

			
			
			fs = FileSystem.get(conf);
			String shifuHome = System.getenv("SHIFU_HOME");
			String packageDir = shifuHome + "/TFDependancy";
			String version = shifuVersion;
			String appMasterJar = "shifu-" + version + ".jar";
			String appMasterJarPath = shifuHome + "/lib" + "/" + appMasterJar;
			//LOG.info("Copy App Master jar from local filesystem and add to local environment");
			addToLocalResources(fs, appMasterJar, appMasterJarPath, localResources);
			//LOG.info("Copy Tensorflow Python Script from local filesystem and add to local environment");
			addToLocalResources(fs, "script.py", "./models/script.py", localResources);
			//LOG.info("Copy python2.7.zip from local filesystem and add to local environment");
			addToLocalResources(fs, "python2.7.zip", packageDir + "/python2.7.zip", localResources);
			//LOG.info("Copy glibc_2.17.zip from local filesystem and add to local environment");
			addToLocalResources(fs, "glibc_2.17.zip", packageDir + "/glibc_2.17.zip", localResources);
			addToLocalResources(fs, "site-package.zip", packageDir + "/site-package.zip", localResources);
			
			
			
			LOG.info("Set the environment for the application master");
			Map<String, String> env = new HashMap<String, String>();
			StringBuilder classPathEnv = new StringBuilder(Environment.CLASSPATH.$$())
			  .append(ApplicationConstants.CLASS_PATH_SEPARATOR).append("./*");
			for (String c : conf.getStrings(
			    YarnConfiguration.YARN_APPLICATION_CLASSPATH,
			    YarnConfiguration.DEFAULT_YARN_CROSS_PLATFORM_APPLICATION_CLASSPATH)) {
			  classPathEnv.append(ApplicationConstants.CLASS_PATH_SEPARATOR);
			  classPathEnv.append(c.trim());
			}
			if (conf.getBoolean(YarnConfiguration.IS_MINI_YARN_CLUSTER, false)) {
				classPathEnv.append(':');
				classPathEnv.append(System.getProperty("java.class.path"));
			}
			env.put(Environment.CLASSPATH.name(), classPathEnv.toString());
			//for(Entry<>)
			
			env.put("DATA_PATH_DIR", dataPathDir);
			env.put("SHIFU_VERSION", shifuVersion);
			env.put("MAX_WAIT_TIME_SYN", "300000");
			env.put("MAX_WAIT_TIME_AM", "10800000");
			env.put("CONTAINER_MEMORY", "1024");
			env.put("CONTAINER_VIRTUAL_CORES", "1");
			env.put("REGEX", "10\\.\\d+?\\.\\d+?\\.\\d+?");
			try {
				env.putAll(JSONUtils.readValue(shifuHome+"/conf/yarnConf.json", HashMap.class));
				//env.putAll(new ObjectMapper().readValue(shifuHome+"/conf/yarnConf.json", Map<String, String>));
			}catch(Exception e) {
				e.printStackTrace();
			}
			LOG.info("Setting up app master command");
			//vargs.add(Environment.JAVA_HOME.$$() + "/bin/java");
			// Set Xmx based on am memory size
			int amMemory = 1024;
			int amVCores = 2;
			int amPriority = 0;
			String amQueue = "default";
			//vargs.add("-Xmx" + amMemory + "m");
			// Set class name
			//vargs.add(ApplicationMaster.class.getName());
			// Set params for Application Master
			List<String> commands = new LinkedList<String>();
			StringBuilder command = new StringBuilder();
			command.append(Environment.JAVA_HOME.$$()).append("/bin/java  ");
			command.append("-Dlog4j.configuration=container-log4j.properties ");
			command.append("-Dyarn.app.container.log.dir=" + 
					ApplicationConstants.LOG_DIR_EXPANSION_VAR + "aaa ");
			command.append("-Dyarn.app.container.log.filesize=0 ");
			command.append("-Dhadoop.root.logger=INFO,CLA ");
			command.append(String.format("ml.shifu.yarn.ApplicationMaster %s %s ", dataPathDir, shifuVersion));
			command.append("1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout ");
			command.append("2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stderr ");
			commands.add(command.toString());
		
			ContainerLaunchContext amContainer = ContainerLaunchContext
					.newInstance(localResources, env, commands, null, null, null);
			
			LOG.info("Completed setting up app master command " + command.toString());
			
			
			Resource capability = Resource.newInstance(amMemory, amVCores);
			appContext.setResource(capability);

			
			LOG.info("Trying to get token...");
			if (UserGroupInformation.isSecurityEnabled()) {
			  // Note: Credentials class is marked as LimitedPrivate for HDFS and MapReduce
			  Credentials credentials = new Credentials();
			  String tokenRenewer = conf.get(YarnConfiguration.RM_PRINCIPAL);
			  if (tokenRenewer == null || tokenRenewer.length() == 0) {
			    throw new IOException(
			      "Can't get Master Kerberos principal for the RM to use as renewer");
			  }

			  // For now, only getting tokens for the default file-system.
			  final Token<?> tokens[] =
			      fs.addDelegationTokens(tokenRenewer, credentials);
			  if (tokens != null) {
			    ;
			  }
			  DataOutputBuffer dob = new DataOutputBuffer();
			  credentials.writeTokenStorageToStream(dob);
			  ByteBuffer fsTokens = ByteBuffer.wrap(dob.getData(), 0, dob.getLength());
			  amContainer.setTokens(fsTokens);
			}
			LOG.info("Get token Completely!");
			
			appContext.setAMContainerSpec(amContainer);
			// Set the priority for the application master
			Priority pri = Priority.newInstance(amPriority);
			appContext.setPriority(pri);
			// Set the queue to which this application is to be submitted in the RM
			appContext.setQueue(amQueue);

			// Submit the application to the applications manager
			// SubmitApplicationResponse submitResp = applicationsManager.submitApplication(appRequest);
			LOG.info("Trying to submit Application...");
			yarnClient.submitApplication(appContext);
			LOG.info("Submit Successfully!");
			
			long start = Time.now();
			//ApplicationReport report = yarnClient.getApplicationReport(appId);
			while(!yarnClient.getApplicationReport(appId).getFinalApplicationStatus().name().equalsIgnoreCase("SUCCEEDED") &&
					Time.now()-start < MAX_WAIT_TIME) {
				LOG.info("Task not complete...");
				Thread.sleep(5000);
			}
			LOG.info("Task complete!");
			fs.copyToLocalFile(new Path(fs.getHomeDirectory(),"shifu_tmp/graph.pb"), new Path("./models/graph.pb"));
		}catch(Exception e) {
			e.printStackTrace();
		}finally {
			try {
				//fs.de
				if(!fs.delete(new Path(fs.getHomeDirectory(),"shifu_tmp"),true))
					throw new Exception("Not delete the file under /shifu_tmp...");
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	private static void addToLocalResources(FileSystem fs, String fileName,
			String fileSrcPath, Map<String, LocalResource> localResources)
			throws IllegalArgumentException, IOException {
		String suffix = "shifu_tmp" + "/" + fileName;
		Path dst = new Path(fs.getHomeDirectory(), suffix);
		LOG.info("hdfs copyFromLocalFile " + fileSrcPath + " =>" + dst);
		fs.copyFromLocalFile(new Path(fileSrcPath), dst);
		FileStatus scFileStatus = fs.getFileStatus(dst);
		LocalResource scRsrc;
		if(fileSrcPath.endsWith(".zip"))
			scRsrc = LocalResource.newInstance(
					ConverterUtils.getYarnUrlFromPath(dst), LocalResourceType.ARCHIVE,
					LocalResourceVisibility.APPLICATION, scFileStatus.getLen(),
					scFileStatus.getModificationTime());
		else
			scRsrc = LocalResource.newInstance(
				ConverterUtils.getYarnUrlFromPath(dst), LocalResourceType.FILE,
				LocalResourceVisibility.APPLICATION, scFileStatus.getLen(),
				scFileStatus.getModificationTime());
		localResources.put(fileName, scRsrc);
	}
	
	public static void main(String[] args) {
		try {
//			new Client().run();
		}catch(Exception e) {}
		
	}
}
