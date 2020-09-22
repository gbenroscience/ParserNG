/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author GBEMIRO
 */
public class Test {
    
    
    
    public static void main(String[]args){
        
        class Box{
            int width;
            int height;

            public Box(int width, int height) {
                this.width = width;
                this.height = height;
            }

            public void setHeight(int height) {
                this.height = height;
            }

            public void setWidth(int width) {
                this.width = width;
            }

            public int getHeight() {
                return height;
            }

            public int getWidth() {
                return width;
            }

            @Override
            public String toString() {
                return "Box width = "+width+", height = "+height+" id = "+super.toString();
            }
            
            
            
        }
        
        ArrayList<Box>boxes = new ArrayList<Box>();
        
        boxes.add(new Box(2,3));
        boxes.add(new Box(4,30));
        boxes.add(new Box(22,32));
        boxes.add(new Box(41,51));
        boxes.add(new Box(42,51));
        boxes.add(new Box(43,51));
        boxes.add(new Box(44,51));
        boxes.add(new Box(45,51));
        boxes.add(new Box(46,51));
        boxes.add(new Box(47,51));
        boxes.add(new Box(48,51));
        boxes.add(new Box(49,51));
        boxes.add(new Box(50,51));
        
        List<Box>boxers = boxes.subList(4, boxes.size());
     for(Box box:boxers){   
    	  System.out.println("The box is "+box);
     }

        
    }//end method
}//end class