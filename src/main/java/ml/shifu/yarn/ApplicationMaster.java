package ml.shifu.yarn;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;

import org.apache.hadoop.util.Time;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.yarn.client.api.async.impl.NMClientAsyncImpl;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.apache.hadoop.yarn.util.ConverterUtils;


//import java.nio.file.*;

public class ApplicationMaster{
	//private static final Logger LOG = LoggerFactory.getLogger(ApplicationMaster.class);
	@SuppressWarnings("rawtypes")
	AMRMClientAsync amRMClient;
	NMClientAsyncImpl nmClientAsync;
	Configuration conf = new Configuration();
	Map<String, String> envs = System.getenv();
	Map<ContainerId, Container> runningContainers = new ConcurrentHashMap<ContainerId, Container>();
	
	String dataPathDir;
	String writeDir;
	int numTotalContainers;
	
	AtomicInteger numCompletedContainers = new AtomicInteger(0);
	ExecutorService exeService = Executors.newCachedThreadPool();
	
	String shifuVersion;
	long MAX_WAIT_TIME;
	FileSystem fs;
	Map<String, LocalResource> localResources = new HashMap<String, LocalResource>();
	Map<String, String> env = new HashMap<String, String>();;
	Credentials credentials = null;
	private ByteBuffer setupTokens() {
        try {
            DataOutputBuffer dob = new DataOutputBuffer();
            credentials.writeTokenStorageToStream(dob);
            return ByteBuffer.wrap(dob.getData(), 0, dob.getLength()).duplicate();
        } catch (IOException e) {
            throw new RuntimeException(e);  // TODO: FIXME
        }
    }
	public void run() throws YarnException, IOException {
		try {
			fs = FileSystem.get(conf);
			System.out.println("123");
			for(String key : envs.keySet())
				System.out.println(key);
			
			dataPathDir = envs.get("DATA_PATH_DIR");
			writeDir = fs.getHomeDirectory().toString() + "/shifu_tmp";
			numTotalContainers = getContainerNum(dataPathDir);
			try {
				numTotalContainers = Integer.valueOf(envs.get("NUM_TOTAL_CONTAINERS"));
			}catch(Exception e){e.printStackTrace();}
			//numTotalContainers = 10;
			shifuVersion = envs.get("SHIFU_VERSION");
			MAX_WAIT_TIME = Long.valueOf(envs.get("MAX_WAIT_TIME_AM"));
			int containerMemory = Integer.valueOf(envs.get("CONTAINER_MEMORY"));
			int containerVirtualCores = Integer.valueOf(envs.get("CONTAINER_VIRTUAL_CORES"));
			String appMasterJar = "shifu-" + shifuVersion + ".jar";
					
			AMRMClientAsync.CallbackHandler allocListener = new RMCallbackHandler();
			amRMClient = AMRMClientAsync.createAMRMClientAsync(1000, allocListener);
			Configuration conf = new Configuration();
			amRMClient.init(conf);
			amRMClient.start();
	
			
			nmClientAsync = new NMClientAsyncImpl(new NMCallbackHandler());
			nmClientAsync.init(conf);
			nmClientAsync.start(); 
			
			credentials = UserGroupInformation.getCurrentUser().getCredentials();
	
			
			
			
			//localResources = 
			addToLocalResources(fs, appMasterJar, appMasterJar, "",
				    localResources);
			addToLocalResources(fs, "script.py", "script.py", "",
				    localResources);
			addToLocalResources(fs, "python2.7", "python2.7.zip", "",
				    localResources);
			addToLocalResources(fs, "glibc_2.17", "glibc_2.17.zip", "",
				    localResources);
			addToLocalResources(fs, "site-package", "site-package.zip", "",
				    localResources);
			
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
			env.put("WRITE_DIR", writeDir);
			env.put("DATA_DIR_PATH", dataPathDir);
			env.put("MAX_WAIT_TIME_SYN", envs.get("MAX_WAIT_TIME_SYN"));
			env.put("REGEX", envs.get("REGEX"));
			env.put("TOTAL_CONTAINER_NUMS", new Integer(numTotalContainers).toString());
			RegisterApplicationMasterResponse response = amRMClient
					.registerApplicationMaster(NetUtils.getHostname(), -1, "");
			
			int maxMem = response.getMaximumResourceCapability().getMemory();
			System.out.println("Max mem capability of resources in this cluster " + maxMem);
	
			int maxVCores = response.getMaximumResourceCapability().getVirtualCores();
			System.out.println("Max vcores capability of resources in this cluster " + maxVCores);
			
			
			// A resource ask cannot exceed the max.
			if (containerMemory  > maxMem) {
			  System.out.println("Container memory specified above max threshold of cluster."
			      + " Using max value." + ", specified=" + containerMemory + ", max="
			      + maxMem);
			  containerMemory = maxMem;
			}
	
			
			if (containerVirtualCores  > maxVCores) {
			  System.out.println("Container virtual cores specified above max threshold of  cluster."
			    + " Using max value." + ", specified=" + containerVirtualCores + ", max="
			    + maxVCores);
			  containerVirtualCores = maxVCores;
			}
			
			List<Container> previousAMRunningContainers =
			    response.getContainersFromPreviousAttempts();
			System.out.println("Received " + previousAMRunningContainers.size()
			        + " previous AM's running containers on AM registration.");
	
			
			int numTotalContainersToRequest =
			    numTotalContainers - previousAMRunningContainers.size();
			
			for (int i = 0; i < numTotalContainersToRequest; ++i) {
			  ContainerRequest containerAsk = setupContainerAskForRM(containerMemory, containerVirtualCores);
			  amRMClient.addContainerRequest(containerAsk);
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	private ContainerRequest setupContainerAskForRM(int containerMemory, int containerVirtualCores){
		  // setup requirements for hosts
		  // using * as any host will do for the distributed shell app
		  // set the priority for the request
		  int requestPriority = 0;
		  Priority pri = Priority.newInstance(requestPriority);

		//  int containerMemory = 100;
		//int containerVirtualCores = 1;
		// Set up resource type requirements
		  // For now, memory and CPU are supported so we set memory and cpu requirements
		  Resource capability = Resource.newInstance(containerMemory,
		    containerVirtualCores );

		  ContainerRequest request = new ContainerRequest(capability, null, null,
		      pri);
		  System.out.println("Requested container ask: " + request.toString());
		  return request;
		}
	
	public float getProgress() {
		  // set progress to deliver to RM on next heartbeat
		  float progress = (float) numCompletedContainers.get()
		      / numTotalContainers;
		  return progress;
		}
	
	private class NMCallbackHandler implements NMClientAsync.CallbackHandler {

		public void onContainerStarted(ContainerId containerId,
				Map<String, ByteBuffer> allServiceResponse) {
			System.out.println("Container Start Successfully " + containerId.toString());
			
		}
	
		public void onContainerStatusReceived(ContainerId containerId,
				ContainerStatus containerStatus) {
	
		}
	
		public void onContainerStopped(ContainerId containerId) {
			// TODO Auto-generated method stub
			System.out.println("Container Stopped " + containerId.toString());
		}
	
		public void onStartContainerError(ContainerId containerId, Throwable t) {
			// TODO Auto-generated method stub
			System.out.println("Container Occurs Error While Starting " + containerId.toString());
		}
	
		public void onGetContainerStatusError(ContainerId containerId,
				Throwable t) {
			// TODO Auto-generated method stub
	
		}
	
		public void onStopContainerError(ContainerId containerId, Throwable t) {
			// TODO Auto-generated method stub
	
		}

	}
	
	private class RMCallbackHandler implements AMRMClientAsync.CallbackHandler {
		
		public void onContainersCompleted(List<ContainerStatus> statuses) {
			for (ContainerStatus status : statuses) {
				ContainerId id = status.getContainerId();
				System.out.println("Container Completed: " + id.toString() 
						+ " exitStatus="+ status.getExitStatus());
				if (status.getExitStatus() != 0) {
					// restart
				}else {			
					runningContainers.remove(id);
					numCompletedContainers.addAndGet(1);
				}
			}
			if(numCompletedContainers.get() == numTotalContainers-1) {
				try {
					amRMClient.unregisterApplicationMaster(FinalApplicationStatus.SUCCEEDED, "dummy Message", null);
				} catch (YarnException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				amRMClient.stop();
			}
		}
		public void onContainersAllocated(List<Container> containers) {
			for (Container c : containers) {
				System.out.println("Container Allocated"
						+ ", id=" + c.getId() 
						+ ", containerNode=" + c.getNodeId());
				exeService.submit(new LaunchContainerTask(c));
				runningContainers.put(c.getId(), c);
			}
		}

		public void onShutdownRequest() {
		}

		public void onNodesUpdated(List<NodeReport> updatedNodes) {

		}

		public float getProgress() {
			float progress = 0;
			return progress;
		}

		public void onError(Throwable e) {
			amRMClient.stop();
			e.printStackTrace();
		}

	}
	private class LaunchContainerTask implements Runnable {
		Container container;
		
		public LaunchContainerTask(Container container) {
			this.container = container;
		}
		public void run(){
			try {
				List<String> commands = new LinkedList<String>();
				StringBuilder command1 = new StringBuilder();
				command1.append(Environment.JAVA_HOME.$$()).append("/bin/java  ");
				command1.append("-Dlog4j.configuration=container-log4j.properties ");
				command1.append("-Dyarn.app.container.log.dir=" + 
						ApplicationConstants.LOG_DIR_EXPANSION_VAR + "aaa ");
				command1.append("-Dyarn.app.container.log.filesize=0 ");
				command1.append("-Dhadoop.root.logger=INFO,CLA ");
				command1.append("ml.shifu.yarn.SynWorker ");
				command1.append("1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout ");
				command1.append("2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stderr;");			
				commands.add(command1.toString());
				
				commands.add("export PYTHONPATH=./site-package/site-packages/:$PYTHONPATH;");
				
				StringBuilder command2 = new StringBuilder();
				command2.append("./glibc_2.17/glibc2.17/lib/ld-linux-x86-64.so.2 ");
				command2.append("--library-path ./glibc_2.17/glibc2.17/lib/:$LD_LIBRARY_PATH:/usr/lib64 ");
				command2.append("./python2.7/python2.7/bin/python ");
				command2.append("./script.py ");
				command2.append(String.format("--node_nums=%d ", numTotalContainers));		
				command2.append("1>>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout ");
				command2.append("2>>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stderr;");
				commands.add(command2.toString());
				commands.add("ls -R ./tmp;");
				commands.add("hadoop fs -put ./tmp/train_logs/*.pbtxt " + fs.getHomeDirectory()+"/shifu_tmp/");
				ContainerLaunchContext ctx = ContainerLaunchContext.newInstance(
						localResources, env, commands, null, null, null);
				ctx.setTokens(setupTokens());
				nmClientAsync.startContainerAsync(container, ctx);
			}
			catch(Throwable e) {
				e.printStackTrace();
			}
		}
	}
	private void addToLocalResources(FileSystem fs, String fileName,
			String srcFileName, String appId,
			Map<String, LocalResource> localResources)
			throws IllegalArgumentException, IOException {
		String suffix = writeDir + "/" + srcFileName;
		Path dst = new Path(fs.getHomeDirectory(), suffix);
		System.out.println(dst);
		FileStatus scFileStatus = fs.getFileStatus(dst);
		System.out.println(scFileStatus.getLen());
		LocalResource scRsrc;
		if(srcFileName.endsWith(".zip"))
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
	void waitComplete() throws YarnException, IOException{
		long start = Time.now();
		System.out.println(MAX_WAIT_TIME);
		while(numTotalContainers != numCompletedContainers.get() && (Time.now()-start) < MAX_WAIT_TIME){
			try{
				Thread.sleep(1000);
				
			} catch (InterruptedException ex){}
		}
		System.out.println("ShutDown exeService Start");
		exeService.shutdown();
		System.out.println("ShutDown exeService Complete");
		nmClientAsync.stop();
		System.out.println("amNMClient  stop  Complete");
		amRMClient.unregisterApplicationMaster(FinalApplicationStatus.SUCCEEDED, "dummy Message", null);
		System.out.println("unregisterApplicationMaster  Complete");
		amRMClient.stop();
		System.out.println("amRMClient  stop Complete");
	}
	private int getContainerNum(String dataPathDir) throws Exception {
		try {
			List<FileStatus> dataFileStatus = new ArrayList<FileStatus>();
	    	for (FileStatus status : fs.listStatus(new Path(dataPathDir))){
	    		if (status.getPath().getName().endsWith(".gz"))
	    			dataFileStatus.add(status);
	    	}
	    	int dataFileNums = dataFileStatus.size();
	    	return dataFileNums <= 40 ? dataFileNums+1 : 41;
		}catch(Exception e) {
			throw e;
		}
	}
	public static void main(String[] args) {
		ApplicationMaster am = null;
		try {
			am = new ApplicationMaster();
			//am.dataPathDir = args[0];
			//am.shifuVersion = args[1];
			am.run();
			am.waitComplete();
		} catch (YarnException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {
			
		}
	}
}

