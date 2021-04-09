/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;
 
 

import interfaces.Savable;
import java.io.StringReader;
import java.util.InputMismatchException;

import math.Maths;
import java.util.NoSuchElementException;
import util.Serializer;

/**
 *
 * @author GBENRO
 *
 * template for modeling/creating variable objects
 */
public class Variable implements Savable{

//a String variable that represents the String property of the varriable
    private String name;

    private TYPE type;

    //the value stored by this variable
    private String value;
    //the units stored by this variable..not compulsory
    private String units = "";

    private boolean constant;//if this value is true,then the Variable object is not modifiable, i.e it represents a constant.
    /**
     * The full name of the variable
     */
    private String fullName;

    /**
     * The constant PI
     */
    public static final Variable PI = new Variable("pi", Maths.PI(), true);

    /**
     * The last answer variable used for flexibility on computational systems.
     */
    public static Variable ans = new Variable("ans", "0.0", false);

    public static Variable e = new Variable("e", String.valueOf(Math.E), true);

    static {
        ans.type = TYPE.NUMBER;
        e.type = TYPE.NUMBER;
        PI.type = TYPE.NUMBER;
    }

    public Variable(String name) {

        if (isVariableString(name)) {
            this.name = name;
            this.value = "0.0";
            this.type = TYPE.NUMBER;
        } else {

        }
    }

    /**
     *
     * @param name the name of the Variable object e.g A,B...e.t.c
     * @param value the value stored by the Variable object
     * @param constant the nature of the Variable object whether it is
     * modifiable or not. If constant = true , then the Variable object
     * represents a constant, whose value cannot be altered.Else,it represents a
     * Variable object whose value can change.
     */
    public Variable(String name, String value, boolean constant) {
        this(name, "", value, constant);
    }

    /**
     *
     * @param name the name of the Variable object e.g A,B...e.t.c
     * @param fullName the full name of the Variable object
     * @param value the value stored by the Variable object
     *
     * @param constant the nature of the Variable object whether it is
     * modifiable or not. If constant = true , then the Variable object
     * represents a constant, whose value cannot be altered.Else,it represents a
     * Variable object whose value can change.
     */
    public Variable(String name, String fullName, String value, boolean constant) {

        if (isVariableString(name)) {
            this.name = name;
            setValue(value);
            this.fullName = fullName;
            this.constant = constant;
        }

    }

    public void setType(TYPE type) {
        this.type = type;
    }

    public TYPE getType() {
        return type;
    }

    /**
     *
     * @param name the name of the Variable object e.g A,B...e.t.c
     * @param value the value stored by the Variable object
     * @param constant the nature of the Variable object whether it is
     * modifiable or not. If constant = true , then the Variable object
     * represents a constant, whose value cannot be altered.Else,it represents a
     * Variable object whose value can change.
     */
    public Variable(String name, double value, boolean constant) {
        this(name, value + "", constant);
    }

    public void setFullName(String fullName) {
        this.fullName = fullName;
    }

    public String getFullName() {
        return fullName == null || fullName.isEmpty() ? name: fullName;
    }

    /**
     *
     * @return true if the Variable object is a constant.
     */
    public boolean isConstant() {
        return constant;
    }

    /**
     * sets the nature of the Variable object to either Variable or constant
     *
     * @param constant = true if the Variable object is to be changed into a
     * constant and constant = false if the Variable object is to be changed
     * into a Variable object
     */
    public void setConstant(boolean constant) {
        this.constant = constant;
    }

    public boolean isTheta() {
        return this.getName().equals("θ");
    }

    /**
     *
     * @param var
     * @return true if the variable is a valid Variable starting character
     */
    public static boolean isVariableBeginner(String var) {
        return (!Operator.isPermOrComb(var) && var.charAt(0) != 'E' && var.charAt(0) != 'π'
                && (Character.isLetter(var.charAt(0)) || isTheta(var) || isPI(var) || isExpNumber(var) || var.equals("_") || var.equals("$")));
    }//end method

    /**
     * If the name is a string of alphabets and is not an operator name, then it
     * is a valid variable name.
     *
     * If the name is alphanumeric,it is a valid variable name.
     *
     *
     * @param var the string to check.
     * @return true if the variable is a valid Variable object name.
     */
    public static boolean isVariableString(String var) {

        if (var.length() == 1) {
            return isVariableBeginner(var);
        } else if (var.length() > 1) {
            if (var.equals("ans")) {
                return true;
            }//end if
            else if (var.equals("var") || var.equals("const") || STRING.purifier(var).equals("")) {
                return false;
            }//end else if
            else if (Operator.isOperatorString(var)) {
                return false;
            }//end else if

            if (isVariableBeginner(var.substring(0, 1))) {

                int sz = var.length();

                for (int i = 0; i < sz; i++) {
                    if (!isVariableBuilder(var.substring(i, i + 1))) {
                        return false;
                    }//end if
                }//end for loop

                return true;
            }//end if
            else {
                return false;
            }//end else

        }//end else if

        return false;
    }//end method.

    public static boolean isVariableBuilder(String unit) {
        return Character.isLetterOrDigit(unit.charAt(0)) || isTheta(unit)
                || isPI(unit) || isExpNumber(unit) || unit.equals("_") || unit.equals("$");
    }

    /**
     *
     * @param str the name of the String variable
     * @return true if the variable is one that is already defined by the parser
     * for its own purposes. An example of such a variable is any constant
     * parameter recognized by the parser, i.e to which the parser has attached
     * a meaning already. Also any parameter such as the "ans" parameter which
     * returns the last value calculated by the parser is regarded by the parser
     * to be a system variable.
     *
     */
    public static boolean isSystemVar(String str) {
        return str.equals(ans.name) || str.equals(e.name) || str.equals(PI.name);
    }

    /**
     *
     * @param str the name of the String variable
     * @return true if the variable is a constant one that is already defined by
     * the parser for its own purposes. An example of such a constant is any
     * constant defined by the parser e.g PI and so on.
     *
     */
    public static boolean isSystemConstant(String str) {
        return str.equals("π") || str.equals("e");
    }

    /**
     *
     * @param name the name of the system constant
     * @return the constant value associated with it.
     */
    public static String getSystemConstantValue(String name) {
        if (isPI(name)) {
            return Maths.PI();
        }//end if
        else if (isExpNumber(name)) {
            return Variable.e.getName();
        }//end else if
        throw new InputMismatchException("Not System Constant");
    }//end method

    /**
     *
     * @param str the name of the String variable
     * @return true if the variable is ë
     *
     */
    public static boolean isExpNumber(String str) {
        return str.equals("e");
    }

    /**
     *
     * @param str the name of the String variable
     * @return true if the variable is PI
     *
     */
    public static boolean isPI(String str) {
        return str.equals("pi");
    }

    /**
     *
     * @param str the name of the String variable
     * @return true if the variable is theta
     *
     */
    public static boolean isTheta(String str) {
        return str.equals("θ");
    }

    /**
     *
     * @param str the name of the String variable
     * @return true if the variable is the last result evaluated
     *
     */
    public static boolean isLastEvaluatedAnswer(String str) {
        return str.equals("ans");
    }

    /**
     *
     * @return true if the variable is a simple one i.e A-Z or theta
     */
    public boolean isSimpleVar() {
        return ((this.name.length() == 1) && (STRING.isUpperCaseWord(name) || name.equals("θ")));
    }

    /**
     *
     * @param name mutator method that changes the name of the variable
     */
    public void setName(String name) {
        if (isVariableString(name)) {
            this.name = name;
        }

    }

    /**
     *
     * @return the String property of the variable
     */
    public String getName() {
        return name;
    }

    /**
     * changes the value stored in the variable
     *
     * @param value The value to be stored. Sometimes the value contains a unit.
     * To support this, the user should place a space between the value and the
     * units.
     *
     */
    public void setValue(String value) {

        if (value.contains(" ")) {
            String[] vals = value.split(" ");
            value = vals[0];
            this.units = vals[1];
        }

        if (Number.validNumber(value)) {
            this.value = value;
        } else {
            throw new NumberFormatException("Bad Value! " + value + ".\nVariables store only valid real numbers.");
        }

    }

    /**
     * returns the value of the variable
     *
     * @return the value stored in the variable
     */
    public String getValue() {
        if (isPI(getName())) {
            return value = Maths.PI();
        } else if (isLastEvaluatedAnswer(getName())) {
            return value = MathExpression.lastResult;
        } else if (isExpNumber(getName())) {
            return value;
        } else if (isConstant()) {
            return value;
        }
        return value;
    }

    public void setUnits(String units) {
        this.units = units;
    }

    public String getUnits() {
        return units;
    }

    //
    /**
     *
     * @param var the String property of the variable.This method returns the
     * index of the Variable object that has this String property.
     * @return The index of the Variable object that has this String property.
     */
    public static int getSimpleVarIndex(String var) {
        int index = 0;
        if (var.equals("A")) {
            index = 0;
        } else if (var.equals("B")) {
            index = 1;
        } else if (var.equals("C")) {
            index = 2;
        } else if (var.equals("D")) {
            index = 3;
        } else if (var.equals("E")) {
            index = 4;
        } else if (var.equals("F")) {
            index = 5;
        } else if (var.equals("G")) {
            index = 6;
        } else if (var.equals("H")) {
            index = 7;
        } else if (var.equals("I")) {
            index = 8;
        } else if (var.equals("J")) {
            index = 9;
        } else if (var.equals("K")) {
            index = 10;
        } else if (var.equals("L")) {
            index = 11;
        } else if (var.equals("M")) {
            index = 12;
        } else if (var.equals("N")) {
            index = 13;
        } else if (var.equals("O")) {
            index = 14;
        } else if (var.equals("P")) {
            index = 15;
        } else if (var.equals("Q")) {
            index = 16;
        } else if (var.equals("R")) {
            index = 17;
        } else if (var.equals("S")) {
            index = 18;
        } else if (var.equals("T")) {
            index = 19;
        } else if (var.equals("U")) {
            index = 20;
        } else if (var.equals("V")) {
            index = 21;
        } else if (var.equals("W")) {
            index = 22;
        } else if (var.equals("X")) {
            index = 23;
        } else if (var.equals("Y")) {
            index = 24;
        } else if (var.equals("Z")) {
            index = 25;
        } else if (var.equals("θ")) {
            index = 26;
        } else {
            throw new NoSuchElementException("Only A-Z And The Theta Symbol Are Supported.");
        }
        return index;
    }// end method simpleVariableIndex

    /**
     *
     * @return
     */
    public int getSimpleVarIndex() {
        int index = 0;
        if (getName().equals("A")) {
            index = 0;
        } else if (getName().equals("B")) {
            index = 1;
        } else if (getName().equals("C")) {
            index = 2;
        } else if (getName().equals("D")) {
            index = 3;
        } else if (getName().equals("E")) {
            index = 4;
        } else if (getName().equals("F")) {
            index = 5;
        } else if (getName().equals("G")) {
            index = 6;
        } else if (getName().equals("H")) {
            index = 7;
        } else if (getName().equals("I")) {
            index = 8;
        } else if (getName().equals("J")) {
            index = 9;
        } else if (getName().equals("K")) {
            index = 10;
        } else if (getName().equals("L")) {
            index = 11;
        } else if (getName().equals("M")) {
            index = 12;
        } else if (getName().equals("N")) {
            index = 13;
        } else if (getName().equals("O")) {
            index = 14;
        } else if (getName().equals("P")) {
            index = 15;
        } else if (getName().equals("Q")) {
            index = 16;
        } else if (getName().equals("R")) {
            index = 17;
        } else if (getName().equals("S")) {
            index = 18;
        } else if (getName().equals("T")) {
            index = 19;
        } else if (getName().equals("U")) {
            index = 20;
        } else if (getName().equals("V")) {
            index = 21;
        } else if (getName().equals("W")) {
            index = 22;
        } else if (getName().equals("X")) {
            index = 23;
        } else if (getName().equals("Y")) {
            index = 24;
        } else if (getName().equals("Z")) {
            index = 25;
        } else if (getName().equals("θ")) {
            index = 26;
        } else {
            throw new NoSuchElementException("Only A-Z And The Theta Symbol Are Supported.");
        }
        return index;
    }// end method simpleVariableIndex

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Variable) {
            return this.getName().equals(((Variable) obj).getName());
        }//end if
        else {
            return false;
        }//end else
    }//end method
    
/**
 * 
 * @param enc The encoded format of the byte array: [num1, num2, num3, num4, ...]
 * @return the Variable object that represents the encoded data
 */
    public static Variable parse(String enc) {
        return (Variable) Serializer.deserialize(enc);
    }

    @Override
    public String serialize() {
        return Serializer.serialize(this);
    }

    @Override
    public String toString() {
        return this.name + ":" + this.value;
    }

    /**
     *
     * @param args
     */
    public static void main(String args[]) {
        System.out.println(isVariableString("sin"));
    }//end method

}//end class Variable
