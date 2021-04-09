/*
 * Copyright 2021 GBEMIRO JIBOYE <gbenroscience@gmail.com>.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package util;

import interfaces.Savable;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import parser.CustomScanner;

/**
 *
 * @author GBEMIRO JIBOYE <gbenroscience@gmail.com>
 */
public class Serializer {

  
    
    /**
     * 
     * @param yourObject Serializes the object to a byte array and returns it as a string printed as: [num1, num2, num3, num4, .....]
     * @return a string representation for the byte array representation of the object.
     */
    public static final String serialize(Savable yourObject) {
     return stringifyBytes(ser(yourObject));
    }
    
    public static final Savable deserialize(String encoded) {
          byte[] data = getBytes(encoded); 
          return (Savable) deSer(data);
    }

    /**
     * 
     * @param ser The serialized format for the byte array of the object
     * @return the byte array from the serialized format of the object
     */
    public static final byte[] getBytes(String ser) {
         CustomScanner cs = new CustomScanner(ser , false, "[" ,"]", "," );
        List<String> list = cs.scan();
        
        
        byte[]data = new byte[list.size()];
        
        int i=0;
        for(String txt : list){
            data[i] = Byte.valueOf(txt);
            i++;
        }
        
        return data;
    }

      /**
     * Prepare the byte array
     *
     * @param yourObject
     * @return
     */
    public static final byte[] ser(Savable yourObject) {

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream out = null;
        try {
            out = new ObjectOutputStream(bos);
            out.writeObject(yourObject);
            out.flush();
            return bos.toByteArray();

        } catch (Exception e) {
            return null;
        } finally {
            try {
                bos.close();
            } catch (IOException ex) {
                // ignore close exception
            }
        }

    }
    
    /**
     * Create an object from a byte array:
     *
     * @param yourBytes
     * @return
     */
    private static final Object deSer(byte[] yourBytes) {

        ByteArrayInputStream bis = new ByteArrayInputStream(yourBytes);
        ObjectInput in = null;

        try {
            in = new ObjectInputStream(bis);
            return in.readObject();

        } catch (Exception e) {
            return null;
        } finally {
            try {
                if (in != null) {
                    in.close();
                }
            } catch (IOException ex) {
                // ignore close exception
            }
        }
    }
    
    
    
    
    private static String stringifyBytes(byte[]data){
        StringBuilder builder = new StringBuilder("[");
        int i=0;
        int len = data.length;
        for(byte b : data){
            if(i!=len-1){
                       builder.append(b).append(", ");   
            }else{
                    builder.append(b).append("]");   
            }
            i++;
        }
        return builder.toString();
    }
    
 
    
    public static void main(String[] args) {
        
        class Box implements Savable{
            String name;
            int len;
            int brd;
            int hei;

            public Box(String name, int len, int brd, int hei) {
                this.name = name;
                this.len = len;
                this.brd = brd;
                this.hei = hei;
            }
            
            public String getName(){
                return name;
            }
            
            
            public int getLen(){
                return len;
            }
            
               
            public int getBrd(){
                return brd;
            }
            

            public int getHei() {
                return hei;
            }

            @Override
            public String serialize() {
                return Serializer.serialize(this);
            }
            
            
            
            
            public String toString(){
                return "box-name="+name+", dimensions: [len="+len+", brd="+brd+", hei="+hei+"]";
            }
        }
        
        
        Box bx = new Box("kolo-dollar", 12,13,5);
        
        byte[] data = Serializer.ser(bx);
        
         
 
        Box b = (Box) Serializer.deSer(data);
       
        System.out.println("reconstructed box is: "+b.toString());
       
       String bytes = "[23, 44, 78, 233, 56, 91, 90, 22, 23, -92]";
       
        CustomScanner cs = new CustomScanner(bytes , false, "[" ,"]", "," );
        List<String> list = cs.scan();
        
        System.out.println(list);
        
    }   
}