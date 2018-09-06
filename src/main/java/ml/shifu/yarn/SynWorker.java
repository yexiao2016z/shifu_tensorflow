package ml.shifu.yarn;


import java.io.FileWriter;
import java.io.IOException;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Time;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SynWorker {
	private static final Logger LOG = LoggerFactory.getLogger(SynWorker.class);
	final int START_PORT =  22222;
	final int END_PORT = 40000;
	Map<String, String> envs = System.getenv();
	long MAX_WAIT_TIME;
	String regex;
	Configuration conf = new Configuration();
	FileSystem fs;
	String writeDir;
	String dataDirPath;
	int totalContainerNums;
	public void run() {
		
        @SuppressWarnings("rawtypes")
        String IPAddress = null;
        
        try {
        	fs = FileSystem.get(conf);
        	regex = envs.get("REGEX");
			writeDir = fs.getHomeDirectory().toString() + "/shifu_tmp";
			dataDirPath = envs.get("DATA_DIR_PATH");
			MAX_WAIT_TIME = Long.valueOf(envs.get("MAX_WAIT_TIME_SYN"));
			totalContainerNums =  Integer.valueOf(envs.get("TOTAL_CONTAINER_NUMS"));
			IPAddress =  getIPAddress(regex);
			System.out.println("This communication IPAddress is " + IPAddress);
			System.out.println("Trying write IPAddress File in " + writeDir);
			createFile(writeDir+"/"+IPAddress);
			System.out.println("Successfully write!");
			long startTime = Time.now();
			boolean flag = false;
			System.out.println(String.format("Totally Need %d Awares!",totalContainerNums));
			while(Time.now()-startTime < MAX_WAIT_TIME){
				if(synFile(writeDir, totalContainerNums)) {
					flag = true;
					break;
				}
				try {
					Thread.sleep(5000);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					//System.err.print(e.printStackTrace());
					;
				}
			}
			
			if(!flag) {
				System.err.println("Time out when syn...");
				throw new Exception();
			}
			
		
	        Path path = new Path(writeDir);
	        String allIPAddress;
			
			
			
	        StringBuilder allIPAddressBuilder = new StringBuilder();
	        
	        
	        int task_index = 0;
	        int index = 0;
	    	for (FileStatus status : fs.listStatus(path)) {
	        	String fileName = status.getPath().getName();
	        	System.err.println(fileName);
	        	if(fileName.endsWith(".ip")) {
	        		if(IPAddress.toString().equals(fileName))
	        			task_index= index;
	        		index++;
	        		String[] tmp = fileName.substring(0, fileName.length()-3).split("_");
	        		allIPAddressBuilder.append(String.format("%s.%s.%s.%s:%s,", tmp[0],tmp[1],tmp[2],tmp[3],tmp[4]));
	        		//dataFileStatus.add(status);
	        	}
	        }
	    	allIPAddress = allIPAddressBuilder.toString();
	    	downLoadData(dataDirPath, index, task_index);
		
		
			FileWriter writer=new FileWriter("ipz");
			writer.write(allIPAddress.substring(0,allIPAddress.length()-1));
			writer.close();
        }catch(Exception e) {
    		e.printStackTrace();
    	}
	}
	private void downLoadData(String dataDirPath, int taskNums, int task_index) throws Exception{
    	try {
    		List<FileStatus> dataFileStatus = new ArrayList<FileStatus>();
        	for (FileStatus status : fs.listStatus(new Path(dataDirPath))){
        		if (status.getPath().getName().endsWith(".gz"))
        			dataFileStatus.add(status);
        	}
        	int dataFileNums = dataFileStatus.size();
        	if(task_index >= 0) {
        		//System.err.println(dataFileNums);
        		for(int i = 0; i < dataFileNums / taskNums + 1; i++) {
        			int fileIndex = i*taskNums+task_index;
        			//System.out.println("fileIndex:" + new Integer(fileIndex).toString());
        			//System.out.println(fileIndex);
        			if(fileIndex < dataFileNums) {
        				String fileName = dataFileStatus.get(fileIndex).getPath().getName();
        				System.out.println("Trying copy file " + fileName);
	        			
        				fs.copyToLocalFile(new Path(dataDirPath+fileName), new Path("./"+fileName));
        				System.out.println("Successfully copy file " + fileName);
        			}
        		}
        	}
    	}catch(Exception e) {
    		throw e;
    	}
	}
	
	private String getIPAddress(String regex) throws Exception {
		StringBuilder IPAddress = new StringBuilder();
		try {
			Enumeration netInterfaces = NetworkInterface.getNetworkInterfaces();
			while (netInterfaces.hasMoreElements()) {
		          NetworkInterface ni = (NetworkInterface) netInterfaces.nextElement();
		          //String a = ((InetAddress)ni.getInetAddresses().nextElement()).getHostAddress();
		          System.err.println(ni.getName());
		    	  
		        	  if(ni.getInetAddresses().hasMoreElements()) {
		        		  Enumeration<InetAddress> a = ni.getInetAddresses();
		        		  while(a.hasMoreElements()) {
		        			  InetAddress b = a.nextElement();
		        			  if(Pattern.matches(regex, b.getHostAddress())) {
		        				  String[] tmp = b.getHostAddress().split("\\.");
		        				  IPAddress.append(String.format("%s_%s_%s_%s", tmp[0],tmp[1],tmp[2],tmp[3]));
		        				  break;
		        			  }  
		        		  }
		        	 
		        	  }
			}
		}catch(Exception e) {
			throw e;
		}
		
        if(IPAddress.length() < 1) {
  		  	throw new Exception("Cannot get any IP in partten");
        }
        int port = START_PORT;
        for(; port > END_PORT; port++) {
        	try {
                ServerSocket server = new ServerSocket(port);
                server.close();
                break;
            } catch (IOException e) {
            	continue;
            }
        }
        if(port <= END_PORT) {
        	IPAddress.append(String.format("_%d.ip", port));
        }else {
        	throw new Exception("Cannot get any IP in partten 22222~40000");
        }
        
        return IPAddress.toString();
	}
	
	public static void main(String[] args) throws IllegalArgumentException, IOException {
		
		new SynWorker().run();
	}
	
	private void createFile(String writePath) throws Exception {
		try {
	        Path dstPath = new Path(writePath);
	        FSDataOutputStream outputStream = fs.create(dstPath);
	        
	        outputStream.write("test".getBytes());
	        outputStream.close();
        }catch(Exception e){
        	throw e;
        }
	}
	//private static Set<String> files = new HashSet<String>();
	private boolean synFile(String synDir, int num) throws Exception {
		try {         
            Path path = new Path(synDir);
            int count = 0;
            if (fs.exists(path) && fs.isDirectory(path)) {
                for (FileStatus status : fs.listStatus(path)) {
                	String fileName = status.getPath().toString();
                	if(fileName.endsWith(".ip")) {
                		count++;
                	}
                }
            }else {
            	throw new Exception("No path exists...");
            }
            System.out.println(String.format("Now %d Awares!",count));
            //System.out.println(num);
            if(count == num) {
            	return true;
            }
            return false;	
        } catch (Exception e) {
        	throw e;
        }
	}
}
