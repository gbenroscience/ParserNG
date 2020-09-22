package util.io;

import com.google.gson.Gson;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Map;
import java.io.BufferedWriter;
import java.io.FileWriter;

import com.itis.libs.parserng.android.expressParser.Function;

/**
 *
 * @author JIBOYE, OLUWAGBEMIRO OLAOLUWA on 8/17/2016.
 *
 */
public class FunctionsBackup {






    /**
     * This writes items on a queue to a file.
     * Individual items on the queue ocuupy
     * distinct lines in the file.
     * @param map The map of items
     * @param file The File
     */
    public static void writeMapItemsToFileLineByLine(Map<String, Function> map, File file){
        if(map==null||map.isEmpty()){
            return;
        }

        FileWriter fw = null;
        BufferedWriter bw = null;
        try{

            String newLine = System.getProperty("line.separator");
            fw = new FileWriter(file.getAbsoluteFile());
            bw = new BufferedWriter(fw);

            for(Map.Entry<String,Function>entry:map.entrySet()) {
                Function f = entry.getValue();
                if (f!=null && !f.isAnonymous()){
                    String s = new Gson().toJson(f);
                    bw.write(s + newLine);
            }
            }

            bw.flush();
        } catch (Exception e) {

        }
        finally {
            if(fw!=null){
                try {
                    fw.close();
                }
                catch (Exception e){}
            }
            if(bw!=null){
                try {
                    bw.close();
                }
                catch (Exception e){}
            }
        }

    }

    /**
     * This reads the lines of text in a file into a {@link Map}.
     * Such that individual file lines become individual indexes on the {@link Map}.
     *
     * @param map A {@link Map}
     * @param fileToRead The file
     */
    public static void readFileLinesToMap(Map<String,Function> map, File fileToRead){
        if(map==null){
            return;
        }

        BufferedReader in;
        try {
            FileReader stream = new FileReader(fileToRead);
            in = new BufferedReader( stream );
        }
        catch (Exception e) {
            util.Utils.logError(
                    "Sorry, but an error occurred \nwhile trying to open the file:/n" + e);
            return;
        }
        try {
            while (true) {
                String lineFromFile = in.readLine();
                if (lineFromFile == null){
                    break;  // End-of-file has been reached.
                }
                if(lineFromFile != null && !lineFromFile.equals("null")){
                    Function function = Function.parse(lineFromFile);
                    map.put(function.getName(), function);
                }
            }//end while

        }//end try
        catch (Exception e) {
            util.Utils.logError("Sorry, but an error occurred \nwhile trying to read the data:/n" + e);
        }
        finally {
            if(in!=null) {
                try {
                    in.close();
                } catch (Exception e) {
                }
            }
        }
    }//end read method






}
