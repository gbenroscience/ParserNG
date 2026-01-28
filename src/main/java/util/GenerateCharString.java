/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util;

import java.util.Random;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class GenerateCharString {
/**
 * The item to be generated.
 */
    private String generated="";

    public String getGenerated() {
        return generated;
    }

    public void setGenerated(String generated) {
        this.generated = generated;
    }



    


/**
 *
 * @param n the number of characters in the pin no.
 * @return an n character pin no. string.
 */
public String generate(int n){
String pin="";
Random ran = new Random();

for(int i=0;i<n;i++){
    if(i==0){
        pin+=getVariableStarters(ran.nextInt(31));
    }

    else{
if(ran.nextInt(2)==0){
pin+=getVariableStarters(ran.nextInt(31));
}
else{
 pin+= ran.nextInt(10);
}
    }//end else
}
setGenerated(pin);
return generated;
}








public String getVariableStarters(int j){
 if(j==0){
 return "A";
}
else if(j==1){
 return "B";
}
else if(j==2){
 return "C";
}
else if(j==3){
 return "D";
}
else if(j==4){
 return "E";
}
else if(j==5){
 return "F";
}
else if(j==6){
 return "G";
}
else if(j==7){
 return "H";
}
else if(j==8){
 return "I";
}
else if(j==9){
 return "J";
}
else if(j==10){
 return "K";
}
else if(j==11){
 return "L";
}
else if(j==12){
 return "M";
}
else if(j==13){
 return "N";
}
else if(j==14){
 return "O";
}
else if(j==15){
 return "P";
}
else if(j==16){
 return "Q";
}
else if(j==17){
 return "R";
}
else if(j==18){
 return "S";
}
else if(j==19){
 return "T";
}
else if(j==20){
 return "U";
}
else if(j==21){
 return "V";
}
else if(j==22){
 return "W";
}
else if(j==23){
 return "X";
}
else if(j==24){
 return "Y";
}
else if(j==25){
 return "Z";
}
else if(j==26){
    return "_";
}
else if(j==27){
    return "$";
}
else if(j==28){
   return "#";
}
 else if(j==29){
    return ".";
}
else if(j==30){
    return "π";
}
else if(j==31){
    return "θ";
}
else{
throw new IndexOutOfBoundsException("ENTER A NUMBER BETWEEN 0 AND 31" );
}


}//end method



}
