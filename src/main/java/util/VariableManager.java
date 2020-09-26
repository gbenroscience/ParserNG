/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util;

import parser.CustomScanner;
import parser.MathExpression;
import parser.Operator;
import parser.Variable;

import java.util.*;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class VariableManager {

    public static final String endOfLine = ";";

//for variables that have not being created:
//var x =2;var y=3;var z =4;var u,v,w,x=8;
//const a=1,const b=2;const q,w,e,r,t=9.2; constant cannot be pi or ans.
    //check if the variable already exists.
//if so,the user cannot define it by var x = 2; instead use x =9; to change the value in x.
    public static final Map<String, Variable> VARIABLES = Collections.synchronizedMap(new HashMap<String, Variable>());

    /**
     * Parses commands used to insert and update Variables loaded into the
     * VARIABLES attribute of objects of this class.
     */
    private CommandInterpreter commandParser;

    public VariableManager() {
        VARIABLES.put(Variable.PI.getName(), Variable.PI);
        VARIABLES.put(Variable.ans.getName(), Variable.ans);
        VARIABLES.put(Variable.e.getName(), Variable.e);

        commandParser = new CommandInterpreter();
    }

    public CommandInterpreter getCommandParser() {
        return commandParser;
    }

    public void setCommandParser(CommandInterpreter commandParser) {
        this.commandParser = commandParser;
    }

    public static String getEndOfLine() {
        return endOfLine;
    }

    /**
     *
     * @return the ArrayList object that stores the Variable data of objects of
     * this class.
     */
    public Map<String, Variable> getVarStore() {
        return VARIABLES;
    }

    /**
     * Loads the {@link Variable} objects in this {@link Collection} into the
     * {@link VariableManager#VARIABLES} {@link Map}.
     *
     * @param variables A {@link Collection} of {@link Variable} objects.
     */
    public static void load(Collection<Variable> variables) {
        synchronized (VARIABLES) {

            for (Variable v : variables) {
                VARIABLES.put(v.getName(), v);
            }

        }
    }

    /**
     * Loads the {@link Variable} objects in this {@link Map} into the
     * {@link VariableManager#VARIABLES} {@link Map}.
     *
     * @param variables A {@link Map} of {@link Variable} objects.
     */
    public static void load(Map<String, Variable> variables) {
        load(variables, false);
    }

    /**
     * Loads the {@link Variable} objects in this {@link Map} into the
     * {@link VariableManager#VARIABLES} {@link Map}.
     *
     * @param variables A {@link Map} of {@link Variable} objects.
     * @param clearFirst If true, then the {@link VariableManager#VARIABLES} is
     * cleared first before new content is loaded into it.
     */
    public static void load(Map<String, Variable> variables, boolean clearFirst) {
        synchronized (VARIABLES) {
            if (clearFirst) {
                VARIABLES.clear();
            }

            VARIABLES.putAll(variables);

        }
    }

    /**
     *
     * @param variableName The name attribute of the variable we are searching
     * the variable store for.
     * @return true if it finds a variable by that name in the store.
     */
    public boolean contains(String variableName) {
        return VARIABLES.get(variableName) != null;
    }

    /**
     *
     * Parses a command that creates a single variable or changes its value.
     *
     * @param cmd The command string to parse.
     */
    public Variable parseSingleCommand(String cmd) {
        int indexOfSemiColon = cmd.indexOf(";");

        if (indexOfSemiColon == -1) {
            return null;
        }

        int indexOfEqual = cmd.indexOf("=");

        if (indexOfEqual != -1) {
            String var = cmd.substring(0, indexOfEqual);

            if (Variable.isVariableString(var)) {
                if (FunctionManager.contains(var)) {
                    FunctionManager.delete(var);
                }
                try {
                    String value = new MathExpression(cmd.substring(indexOfEqual + 1)).solve();
                    Variable vv = new Variable(var, value, false);
                    VARIABLES.put(vv.getName(), vv);
                    update();
                    return vv;

                } catch (Exception e) {//handle exceptions
                    return null;
                }
            }
        }

        return null;
    }

    /**
     *
     * Parses a command that creates or changes the value of variables.
     *
     * @param cmd The command string to parse.
     */
    public void parseCommand(String cmd) {
        if (commandParser == null) {
            commandParser = new CommandInterpreter(cmd);
        } else {
            commandParser.setCommand(cmd);
        }
        update();
    }

    /**
     * Saves stored variables and updates the UI that renders the variables.
     */
    public static void update() {

    }

    /**
     * Initializes the variables store and loads them from persistent storage
     */
    public static void init() {

    }

    /**
     *
     * @param vName The name of the Variable object.
     * @return the Variable object that has the name supplied if it exists. If
     * no such Variable object exists, then it returns null.
     */
    public static Variable getVariable(String vName) {
        return VARIABLES.get(vName);
    }//end method

    /**
     * Attempts to retrieve a Variable object from a VariableManager based on
     * its name.
     *
     * @param vName The name of the Variable object.
     * @return the Variable object that has that name or null if the Variable is
     * not found.
     */
    public static Variable lookUp(String vName) {
        return VariableManager.getVariable(vName);
    }//end method

    /**
     * deletes a Variable or constant whose location in VARIABLES is known
     *
     * @param index the index of the Variable object to be deleted
     */
    /**
     * deletes a Variable or constant whose name is known
     *
     * @param varName the name of the Variable object to be deleted
     */
    public static void delete(String varName) {

        VARIABLES.remove(varName);

        update();

    }//end method

    /**
     * Introduces a Variable
     *
     * @param var the name of the Variable object to be added to the Variable
     * Registry
     */
    public static void add(Variable var) {
        if (var != null && !var.isConstant() && !VARIABLES.containsKey(var.getName())) {
            VARIABLES.put(var.getName(), var);
            update();
        }
    }//end method

    /**
     * Introduces an array or variable-args list of Variable
     *
     * @param vars the variable args list of the Variable objects to be added to
     * the Variable Registry
     */
    public static void add(Variable... vars) {

        for (Variable v : vars) {
            if (v != null && !v.isConstant() && !VARIABLES.containsKey(v.getName())) {
                VARIABLES.put(v.getName(), v);
            }
        }

        update();
    }//end method

    /**
     * Clears all Variables
     */
    public static void clearVariables() {

        Iterator<Map.Entry<String, Variable>> it = VARIABLES.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String, Variable> var = it.next();
            if (!var.getValue().isConstant()) {
                it.remove();
            }
        }
        update();
    }

    /**
     * Clears all constants
     */
    public void clearConstants() {

        Iterator<Map.Entry<String, Variable>> it = VARIABLES.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String, Variable> var = it.next();
            if (var.getValue().isConstant()) {
                it.remove();
            }
        }

        update();
    }

    /**
     * Clears All Variables and Constants
     */
    public void clearVariablesAndConstants() {
        VARIABLES.clear();
    }

    /**
     *
     * @param variableNames An array containing valid variable names
     * @return a command string that initializes the variable names to 0.0 e.g
     * if the array is [a,v,b,m,n], then the output is:
     * a=0.0;v=0.0;b=0.0;m=0.0;n=0.0
     */
    public static String generateCommandStringFromVariableNamesArray(String[] variableNames) {
        String cmd = "";
        for (String var : variableNames) {
            cmd = cmd.concat(var.concat("=0.0".concat(VariableManager.endOfLine)));
        }//end for
        return cmd;
    }//end method

    public static final ArrayList<Variable> getVariables() {
        ArrayList<Variable> variables = new ArrayList<>();
        Collection<Variable> vars = VARIABLES.values();
        synchronized (VARIABLES) {
            for (Variable v : vars) {
                if (v != null && !v.isConstant()) {
                    variables.add(v);
                }
            }
        }
        return variables;
    }

    @Override
    public String toString() {
        return "All Variables = " + VARIABLES;
    }

    /**
     * Objects of this class parse a variable initialization or modifying
     * command string. For example var x = 2;var y=3; var z =4;
     * x=9;y=3.123;const k = 3x+z; k=34; var p,q,r=32; const r,s,t=25; z=3.123;
     * All that is needed to execute the parse is to create an object of the
     * class with the available constructor. At the end of the object creation,
     * the parse has been executed and if it is successful, a call to
     * getAccumulator() will return a java.util.List object filled with the
     * Variable objects embedded in the command. Also at the end of the object
     * creation( the call to the constructor of this class) The VARIABLES
     * attribute of the containing class will have been fed with the Variable
     * objects embedded in the command parsed here. The feeding could be the
     * insertion of the Variable objects, if the store doe not already contain
     * them,or updating Variable objects in the store with the incoming ones if
     * they have the same name. The behavior of the parser is such that it
     * interpretes and executes the code on the fly. This means that it will
     * stop inserting data in the VARIABLES only when it detects error in the
     * code.
     *
     *
     */
    public class CommandInterpreter {

        private String command;

        /**
         * At every step of parsing, this boolean records if the expression is
         * valid or not.
         */
        private boolean valid;

        public CommandInterpreter() {

        }

        /**
         *
         * @param command The variable creation command string.
         */
        public CommandInterpreter(String command) {
            setCommand(command);
        }

        public void setCommand(String command) {
            this.command = command.trim();
            this.valid = true;
            parse();
        }

        public String getCommand() {
            return command;
        }

        public void setValid(boolean valid) {
            this.valid = valid;
        }

        public boolean isValid() {
            return valid;
        }

        /**
         * Takes a String object that represents a variable name, checks if the
         * variable exists in the store and then returns the value of the
         * variable.
         *
         * @param expr The math expression to evaluate
         * @return The value of the expression.
         */
        private String getValue(String expr) throws NullPointerException, InputMismatchException {
            MathExpression func = new MathExpression(expr);
            func.setVariableValuesInFunction(func.getScanner());
            func.setDRG(1);
            return func.solve();
        }

        /**
         * A line is a statement that ends with a semicolon that is...';' This
         * method generates lines of instructions from the command that is
         * input.
         *
         * @return an ArrayList object containing all lines of instructions in
         * the input command string.
         */
        private List<String> generateLines() {

            CustomScanner customScanner = new CustomScanner(command, false, endOfLine);
            List<String> lines = customScanner.scan();

            if (command.length() == 0) {
                Utils.logError("No Command Input!");
                setValid(false);
            }//end if
            else {
                if (command.length() > 0 && command.indexOf(endOfLine) == -1) {
                    Utils.logError("End variable initialization or assignment statements with a " + endOfLine + "..\";\"");
                    setValid(false);
                }
                if (!command.endsWith(endOfLine)) {
                    Utils.logError("End variable initialization or assignment statements with a " + endOfLine + "..\";\"");
                    setValid(false);
                }
            }//end else

            return lines;
        }//end method generateLines

        /**
         *
         * @param name The name of the Variable whose type is to be checked.
         * @return true if a constant< A Variable object that is constant> by
         * that name is found in the accumulator of variables during expression
         * parsing or in the variable store. It returns false otherwise. A
         * return value of false means one of two things: 1. The item was found
         * and was found not to be a constant but a variable. 2. The item was
         * not found at all.
         */
        private boolean isConstant(String name) {
            Variable v = VARIABLES.get(name);
            return v == null ? false : v.isConstant();
        }//end method

        /**
         *
         * @param name The name of the Variable whose type is to be checked.
         * @return true if a variable< A Variable object that is not constant>
         * by that name is found in the accumulator of variables during
         * expression parsing or in the variable store. It returns false
         * otherwise. A return value of false means one of two things: 1. The
         * item was found and was found not to be a variable but a constant. 2.
         * The item was not found at all.
         */
        private boolean isVariable(String name) {
            Variable v = VARIABLES.get(name);
            return v == null ? false : !v.isConstant();
        }//end method

        /**
         * Validates a single line of Variable initialization or updating
         * statement.
         *
         * @param line The line.
         * @return true if the line is valid.
         */
        private boolean validateLine(String line) {

//this condition verifies that only one occurrence of
//= is found in the string.
            int ind = line.indexOf("=");
            int lastInd = line.lastIndexOf("=");

            if (ind == lastInd && valid && ind != -1) {

                String part1 = line.substring(0, ind);
                String part2 = line.substring(ind + 1);
                CustomScanner customScanner = new CustomScanner(part1, true, ",", " ");
                List<String> scan = customScanner.scan();
                List<String> whitespaceremover = new ArrayList<String>();
                whitespaceremover.add(" ");
                scan.removeAll(whitespaceremover);
                int sz = scan.size();
                if (sz == 1) {
                    return Variable.isVariableString(scan.get(0));
                }//end if
                else if (sz > 1) {

                    for (int i = 0; i < sz; i++) {
                        try {
                            if ((Variable.isVariableString(scan.get(i)) && !Operator.isComma(scan.get(i + 1)))
                                    || (!Variable.isVariableString(scan.get(i)) && Operator.isComma(scan.get(i + 1)))) {
                                return false;
                            }//end if
                            else if ((Operator.isComma(scan.get(i)) && !Variable.isVariableString(scan.get(i + 1)))
                                    || (!Operator.isComma(scan.get(i)) && Variable.isVariableString(scan.get(i + 1)))) {

                                return false;
                            }//end if
                            if (!Operator.isComma(scan.get(i)) && !Variable.isVariableString(scan.get(i))) {
                                return false;
                            }
                        }//end try
                        catch (IndexOutOfBoundsException boundsException) {
                            break;
                        }
                    }//end for loop
                    return true;
                }//end else if
                else {
                    return false;
                }
            }//end if
            else {
                return false;
            }

        }//end method.

        /**
         * Some scenarios arise here. Scenario: name = value.e.g t=4 Action:
         * Check the variables store to see if a variable by this name exists.
         * If so,the line is correct. But make sure that no line before this one
         * declares it again. e.g store has variable,then var t =4; t=2;.....the
         * t=2; is the scenario now. if the store has declared it then we don't
         * need to declare it before using it again. Else,check if this variable
         * has been declared in a line before this one. If so the line is
         * correct.Else it is wrong,terminate the whole process.
         *
         * Scenario: var|const name = value|expression. var t = 4 or const w=2
         * Action: Check the variables store to see if a variable by this name
         * exists. If it does exist,the line is wrong,terminate the whole
         * process. Else it is correct.
         *
         * Scenario: name1,name2,name3....,nameN = value|expression e.g. var
         * r,a,t,d=32 or const g,h,i,j=3.14 Action: Check the variables store to
         * see if any of these variables have names that match that of any
         * variable in the store. If they do, the code is wrong. Else it is
         * correct if and only if...their exists no line before this one that
         * declares any of the variables in this line.
         *
         * @return true if the line is valid
         */
        private void analyzeLine(String line) {

            setValid(validateLine(line));

            if (isValid()) {
//this condition verifies that only one occurrence of
//= is found in the string.
                int ind = line.indexOf("=");
                int lastInd = line.lastIndexOf("=");

                if (ind == lastInd && valid && ind != -1) {

                    String part1 = line.substring(0, ind);
                    String part2 = line.substring(ind + 1);
                    CustomScanner customScanner = new CustomScanner(part1, false, ",", " ");
                    List<String> scan = customScanner.scan();
                    List<String> whitespaceremover = new ArrayList<String>();
                    whitespaceremover.add(" ");
                    whitespaceremover.add(",");
                    scan.removeAll(whitespaceremover);
                    int sz = scan.size();
                    try {

                        /**
                         * Handle the scenarios: t,w,x,y,z=value|expression or
                         * t=value|expression
                         */
                        if (Variable.isVariableString(scan.get(0))) {

//multiple assignment. e.g  a,v,c,d,r=value|expression...is line format.
                            if (sz > 1) {

                                for (int i = 0; i < sz && valid; i++) {
                                    if (Variable.isVariableString(scan.get(i))) {
                                        try {
                                            boolean variable = isVariable(scan.get(i));
                                            if (variable) {
                                                VARIABLES.put(scan.get(i), new Variable(scan.get(i), getValue(part2), false));
                                            }//end else if
                                            else if (!variable && !contains(scan.get(i))) {
                                                boolean isFunction = FunctionManager.contains(scan.get(i));
                                                if (isFunction) {
                                                    FunctionManager.delete(scan.get(i));
                                                }
                                                VARIABLES.put(scan.get(i), new Variable(scan.get(i), getValue(part2), false));
                                            }//end else if.
                                        }//end try
                                        catch (Exception e) {
                                            e.printStackTrace();
                                            Utils.logError("Syntax Error!!!.\n");
                                            setValid(false);
                                        }
                                    }//end if
                                    else {
                                        Utils.logError("Syntax Error.\n"
                                                + "Check the help menu for valid code\n"
                                                + " to use near  " + scan.get(i) + scan.get(i + 1));
                                        setValid(false);
                                    }//end else
                                }//end for
                            }//end if
                            //single assignment. e.g  a = 4.3....is line format
                            else {

                                if (Variable.isVariableString(scan.get(0))) {
                                    try {
                                        boolean variable = isVariable(scan.get(0));

                                        if (variable) {
                                            VARIABLES.put(scan.get(0), new Variable(scan.get(0), getValue(part2), false));
                                        }//end else if
                                        else if (!variable && !contains(scan.get(0))) {
                                            boolean isFunction = FunctionManager.contains(scan.get(0));
                                            if (isFunction) {
                                                FunctionManager.delete(scan.get(0));
                                            }

                                            VARIABLES.put(scan.get(0), new Variable(scan.get(0), getValue(part2), false));
                                        }//end else if.
                                    }//end try
                                    catch (Exception e) {
                                        e.printStackTrace();
                                        Utils.logError("Syntax Error.11\n");
                                        setValid(false);
                                    }//end catch
                                }//end if
                                else {
                                    Utils.logError("Syntax Error.22\n");
                                    setValid(false);
                                }
                            }//end else

                        }//end if

                    }//end try
                    catch (IndexOutOfBoundsException boundsException) {
                        Utils.logError("Syntax Error.\nPlease Consult The Help File.");
                        setValid(false);
                    }

                }//end if only one equals sign per line.
                else {
                    Utils.logError("Syntax Error. Each line can only \ncontain one assignment operator!\n");
                    setValid(false);
                }//end else

            }//end if

        }//end method

        /**
         * @param name The name of the variable to search for.
         * @return true if the store of variables of objects of the containing
         * class contain a variable or constant having the parameter name.
         */
        private boolean storeHasVariable(String name) {
            return VARIABLES.get(name) != null;
        }

        /**
         * Conducts an holistic parse of the input data.
         */
        public void parse() {
            List<String> lines = generateLines();
            int size = lines.size();

            for (int i = 0; i < size && valid; i++) {
                analyzeLine(lines.get(i));
            }//end for loop
        }//end method

    }//end class CommandInterpreter

    public static void main(String args[]) {
        String cmd = "a,b,c,d,e=20.213;const a1=a; a2=6/(9-5.12424a);a=98.90;const t=a+1/2;b12=sin(a+b)/cos(a+b);"
                + "c1=1/c1;";
        VariableManager manager = new VariableManager();

        manager.parseCommand(cmd);
        System.out.println(" varStore = " + VariableManager.VARIABLES);

        int i = 0;
        while (true) {
            try {
                System.out.println(" Enter Command " + (i + 1));
                String comd = new Scanner(System.in).nextLine();
                manager.parseCommand(comd);
                System.out.println(" varStore = " + VariableManager.VARIABLES);
                ++i;
            }//end try
            catch (Exception e) {
                break;
            }
        }

    }

}
