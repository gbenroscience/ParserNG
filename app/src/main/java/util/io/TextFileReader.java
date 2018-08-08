/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util.io;

import java.io.*;
 

/**
 *
 *@author JIBOYE Oluwagbemiro Olaoluwa
 */
public class TextFileReader {

   private File fileToRead;

    public TextFileReader(File fileToRead) {
        this.fileToRead = fileToRead;
    }

    public TextFileReader(String pathToFileToRead) {
        this.fileToRead = new File(pathToFileToRead);
    }

    public File getFileToRead() {
        return fileToRead;
    }

    public void setFileToRead(File fileToRead) {
        this.fileToRead = fileToRead;
    }

    public void setFileToRead(String fileToRead) {
        this.fileToRead = new File(fileToRead);
    }


 public String read(){
String text = "";
   BufferedReader in;
   try {
      FileReader stream = new FileReader(fileToRead);
      in = new BufferedReader( stream );
   }
   catch (Exception e) {
      util.Utils.logError(
            "Sorry, but an error occurred \nwhile trying to open the file:/n" + e);
      return "";
   }
   try {
      while (true) {
         String lineFromFile = in.readLine();
         if (lineFromFile == null){
            break;  // End-of-file has been reached.
          }
         text = text + lineFromFile;
   }//end while
     }//end try
   catch (Exception e) {
      util.Utils.logError( "Sorry, but an error occurred \nwhile trying to read the data:/n" + e);
   }
   return text;
}//end read method






}