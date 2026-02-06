/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util;

/**
 *
 * @author GBENRO
 *
 * This class combines valid pieces that can be used to construct identifiers of class
 * Variable and uses them to randomly generate names for objects of class Variable
 *
 *
 */

import parser.Variable;
import java.util.Random;
public class VariableGenerator {

/**
 * used to set the state of objects of this class.That is, if or not they will generate a number value randomly
 * to set as its value.
 */
private boolean genValue=false;
/**
 * Creates a new object of this class that can be used to generate Variables randomly.
 * @param n determines the number of characters that the Variable name will have.
 */
/**
 * The Variable object that is initialized randomly.
 */
private Variable var;

/**
 * The Random attribute of this class that allows it to randomize and generate Variable names randomly.
 */
private Random ran=new Random();

    public VariableGenerator(int n){
        String variable="";
        //randomly determine the length of the name of the Variable to be created.
     int num1=ran.nextInt(n);
     //randomly select one piece amongst A-z or θ to be the starting character
     //in the Variable name which is a necessary condition for Variable name formation.
     int num2=ran.nextInt(53);
     variable=decodePiece(num2);
     
  for(int i=0;i<n;i++){
     int num3=ran.nextInt(67);//randomly generate a number to use as code for randomly decoding a valid Variable
      variable+=decodePiece(num3);
  }
var=new Variable(variable,"0.0",true);
     }//end constructor
/**
 *  creates a new object of this class that can be used to generate Variables randomly
 * @param n determines the number of characters that the Variable name will have.
 * @param genValue variable that determines if the client needs the randomly generated variable
 * to have a randomly specified value too.
 */
public VariableGenerator(int n,boolean genValue){
     this(n);
     this.genValue=genValue;

     if(genValue){
         int sign=ran.nextInt(2);//randomly assign a sign to the number generated.
         if(sign==0){
     double num=-1*ran.nextInt(2000000000);//generate a numerator randomly
     double den=ran.nextInt(2000000000);//generate a denominator randomly
       var.setValue(String.valueOf( num/den ));
         }
         else if(sign==1){
     double num=ran.nextInt(2000000000);//generate a numerator randomly
     double den=ran.nextInt(2000000000);//generate a denominator randomly
       var.setValue( (String.valueOf( num/den ))  );
         }

   

     }
}

/**
 *  creates a new object of this class that can be used to generate Variables randomly
 * @param n determines the number of characters that the Variable name will have.
 * @param val client specified value for the new Variable
 * to have a randomly specified value too.
 */
public VariableGenerator(int n,double val){
     this(n);
     genValue=false;
      var.setValue( (String.valueOf( val ))  );
}




/**
 *
 * @return the state of objects of this class that is whether or not they will randomly
 * generate values for Variable objects.
 * 
 */
    public boolean isGenValue() {
        return genValue;
    }
/**
 *sets the state of objects of this class
 * @param genValue the state of the object:that is whether or not they will randomly
 * generate values for Variable objects.
 */

    public void setGenValue(boolean genValue) {
        this.genValue = genValue;
    }

    public Random getRan() {
        return ran;
    }

    public void setRan(Random ran) {
        this.ran = ran;
    }




    public Variable generateVariable(){

        if(Variable.isVariableString( var.getName()) ){

        }
        return var;
    }


    public String generateZeroes(int n){
        String zeroes="";
        int i=0;
        while(i<n){
            zeroes+="0";
        }
return zeroes;
    }//end method generateZeroes
    /**
     *
     * @param aPiece takes a String character that can form a valid part of the name of
     * an object of class Variable and encodes it
     * @return the code of the character
     */
public int codePieces(String aPiece){
    int code=-23;
if( aPiece.equals("A") ){
 code = 0;
}  else if( aPiece.equals("B") ){
 code = 1;
}  else if( aPiece.equals("C") ){
 code = 2;
}  else if( aPiece.equals("D") ){
 code = 3;
}  else if( aPiece.equals("E") ){
 code = 4;
}  else if( aPiece.equals("F") ){
 code = 5;
}  else if( aPiece.equals("G") ){
 code = 6;
}  else if( aPiece.equals("H") ){
 code = 7;
}  else if( aPiece.equals("I") ){
 code = 8;
}  else if( aPiece.equals("J") ){
 code = 9;
}  else if( aPiece.equals("K") ){
 code = 10;
}  else if( aPiece.equals("L") ){
 code = 11;
}  else if( aPiece.equals("M") ){
 code = 12;
}  else if( aPiece.equals("N") ){
 code = 13;
}  else if( aPiece.equals("O") ){
 code = 14;
}  else if( aPiece.equals("P") ){
 code = 15;
}  else if( aPiece.equals("Q") ){
 code = 16;
}  else if( aPiece.equals("R") ){
 code = 17;
}  else if( aPiece.equals("S") ){
 code = 18;
}  else if( aPiece.equals("T") ){
 code = 19;
}  else if( aPiece.equals("U") ){
 code = 20;
}  else if( aPiece.equals("V") ){
 code = 21;
}  else if( aPiece.equals("W") ){
 code = 22;
}  else if( aPiece.equals("X") ){
 code = 23;
}  else if( aPiece.equals("Y") ){
 code = 24;
}  else if( aPiece.equals("Z") ){
 code = 25;
}  else if( aPiece.equals("θ") ){
 code = 26;
}  else if( aPiece.equals("a") ){
 code = 27;
}  else if( aPiece.equals("b") ){
 code = 28;
}  else if( aPiece.equals("c") ){
 code = 29;
}  else if( aPiece.equals("d") ){
 code = 30;
}  else if( aPiece.equals("e") ){
 code = 31;
}  else if( aPiece.equals("f") ){
 code = 32;
}  else if( aPiece.equals("g") ){
 code = 33;
}  else if( aPiece.equals("h") ){
 code = 34;
}  else if( aPiece.equals("i") ){
 code = 35;
}  else if( aPiece.equals("j") ){
 code = 36;
}  else if( aPiece.equals("k") ){
 code = 37;
}  else if( aPiece.equals("l") ){
 code = 38;
}  else if( aPiece.equals("m") ){
 code = 39;
}  else if( aPiece.equals("n") ){
 code = 40;
}  else if( aPiece.equals("o") ){
 code = 41;
}  else if( aPiece.equals("p") ){
 code = 42;
}  else if( aPiece.equals("q") ){
 code = 43;
}  else if( aPiece.equals("r") ){
 code = 44;
}  else if( aPiece.equals("s") ){
 code = 45;
}  else if( aPiece.equals("t") ){
 code = 46;
}  else if( aPiece.equals("u") ){
 code = 47;
}  else if( aPiece.equals("v") ){
 code = 48;
}  else if( aPiece.equals("w") ){
 code = 49;
}  else if( aPiece.equals("x") ){
 code = 50;
}  else if( aPiece.equals("y") ){
 code = 51;
}  else if( aPiece.equals("z") ){
 code = 52;
}  else if( aPiece.equals("0") ){
 code = 53;
}  else if( aPiece.equals("1") ){
 code = 54;
}  else if( aPiece.equals("2") ){
 code = 55;
}  else if( aPiece.equals("3") ){
 code = 56;
}  else if( aPiece.equals("4") ){
 code = 57;
}  else if( aPiece.equals("5") ){
 code = 58;
}  else if( aPiece.equals("6") ){
 code = 59;
}  else if( aPiece.equals("7") ){
 code = 60;
}  else if( aPiece.equals("8") ){
 code = 61;
}  else if( aPiece.equals("9") ){
 code = 62;
}  else if( aPiece.equals("$") ){
 code = 63;
}  else if( aPiece.equals("#") ){
 code = 64;
}  else if( aPiece.equals("_") ){
 code = 65;
}  else if( aPiece.equals(".") ){
 code = 66;
}
    if(code==-23){
        throw new IllegalArgumentException();
    }

return code;
}//end method codePieces

/**
 * @param code the number code that hides a character or symbol that can form part of a Variable object's name property.
 * @return returns the String character associated with a number code
 * where the character is one that can form a part of the name of an object of class Variable
 */
public String decodePiece(int code){
String decoded="";
if(code==0){
decoded="A";
}
else if(code==1){
decoded="B";
}
else if(code==2){
decoded="C";
}
else if(code==3){
decoded="D";
}
else if(code==4){
decoded="E";
}
else if(code==5){
decoded="F";
}
else if(code==6){
decoded="G";
}
else if(code==7){
decoded="H";
}
else if(code==8){
decoded="I";
}
else if(code==9){
decoded="J";
}
else if(code==10){
decoded="K";
}
else if(code==11){
decoded="L";
}
else if(code==12){
decoded="M";
}
else if(code==13){
decoded="N";
}
else if(code==14){
decoded="O";
}
else if(code==15){
decoded="P";
}
else if(code==16){
decoded="Q";
}
else if(code==17){
decoded="R";
}
else if(code==18){
decoded="S";
}
else if(code==19){
decoded="T";
}
else if(code==20){
decoded="U";
}
else if(code==21){
decoded="V";
}
else if(code==22){
decoded="W";
}
else if(code==23){
decoded="X";
}
else if(code==24){
decoded="Y";
}
else if(code==25){
decoded="Z";
}
else if(code==26){
decoded="θ";
}
else if(code==27){
decoded="a";
}
else if(code==28){
decoded="b";
}
else if(code==29){
decoded="c";
}
else if(code==30){
decoded="d";
}
else if(code==31){
decoded="e";
}
else if(code==32){
decoded="f";
}
else if(code==33){
decoded="g";
}
else if(code==34){
decoded="h";
}
else if(code==35){
decoded="i";
}
else if(code==36){
decoded="j";
}
else if(code==37){
decoded="k";
}
else if(code==38){
decoded="l";
}
else if(code==39){
decoded="m";
}
else if(code==40){
decoded="n";
}
else if(code==41){
decoded="o";
}
else if(code==42){
decoded="p";
}
else if(code==43){
decoded="q";
}
else if(code==44){
decoded="r";
}
else if(code==45){
decoded="s";
}
else if(code==46){
decoded="t";
}
else if(code==47){
decoded="u";
}
else if(code==48){
decoded="v";
}
else if(code==49){
decoded="w";
}
else if(code==50){
decoded="x";
}
else if(code==51){
decoded="y";
}
else if(code==52){
decoded="z";
}
else if(code==53){
decoded="0";
}
else if(code==54){
decoded="1";
}
else if(code==55){
decoded="2";
}
else if(code==56){
decoded="3";
}
else if(code==57){
decoded="4";
}
else if(code==58){
decoded="5";
}
else if(code==59){
decoded="6";
}
else if(code==60){
decoded="7";
}
else if(code==61){
decoded="8";
}
else if(code==62){
decoded="9";
}
else if(code==63){
decoded="$";
}
else if(code==64){
decoded="#";
}
else if(code==65){
decoded="_";
}
else if(code==66){
decoded=".";
}

return decoded;
}//end method decodePiece

 





}//end class RandomVariable
