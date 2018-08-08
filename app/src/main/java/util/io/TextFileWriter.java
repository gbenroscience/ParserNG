package util.io;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */



import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author JIBOYE OLuwagbemiro Olaoluwa
 */
public class TextFileWriter implements Serializable{
private ObjectOutputStream output; // outputs data to file

private String filePath;

public TextFileWriter(String filePath){
this.filePath=filePath;
File file=new File(filePath);
if(!file.exists()){
    try{
file.createNewFile();
    }
    catch(IOException ion){
      
    }
}

openFile();
}

/**
 * Opens a file and prepares it for reading.
 */
      private void openFile()
      {
         try // open file
         {
            output = new ObjectOutputStream(
               new FileOutputStream( filePath ) );
         } // end try
         catch ( IOException ioException )
         {
            System.err.println( "Error opening file." );
         } // end catch
      } // end method openFile

/**
 * Writes object data to the file
 * @param obj The data to be written.
 */
public void write(Object obj){
 try{
     output.writeObject(obj);
 }
    catch(IOException ioErr){

    }
close();
}//end method

/**
 * Writes object data to the file
 * @param file The file
 * @param text The data to be written.
 */
public static void writeText(File file,String text){

PrintWriter writer;
        try {
            writer = new PrintWriter(new FileOutputStream(file), true);
            writer.write(text);
            writer.flush();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TextFileWriter.class.getName()).log(Level.SEVERE, null, ex);
        }

}//end method

// close file and terminate application
      private void close()
      {
         try // close file
         {
            if ( output != null )
               output.close();
         } // end try
         catch ( IOException ioException )
         {
            System.err.println( "Error closing file.");
            System.exit( 1 );
        } // end catch
     } // end method closeFile

    public String getFilePath() {
        return filePath;
    }

    public void setFilePath(String filePath) {
        this.filePath = filePath;
        File file=new File(filePath);
if(!file.exists()){
    try{
file.createNewFile();
    }
    catch(IOException ion){

    }
}

openFile();
    }









public static void main( String args[] ){
    TextFileWriter.writeText( new File("Gbemiro's Friends.txt"), "But the word is nigh thee, even the word of faith, which we preach"
            + "that if thou wouldest confess with thy mouth Adonai Yeshua and thou wouldest believe"
            + "in thy heart that God hath raised Him from the dead,thou wilt be saved."
            + "For with the heart man believes unto righteousness and with the mouth confession is made unto salvation.");


}

}