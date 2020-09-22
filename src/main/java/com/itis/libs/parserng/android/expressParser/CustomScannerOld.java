package com.itis.libs.parserng.android.expressParser;



import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 * @since 2011
 *
 * Objects of this class scans a string
 * into tokens based on a list of tokenizer values.
 */
public class CustomScannerOld {
    /**
     * The input to be scanned into tokens.
     */
    private StringBuilder input;
    /**
     * The tokens to be employed in splitting the
     * input.
     */
    private List<String> splittingTokens = new ArrayList<String>();

    /**
     * The index where a
     * splitting token first appears in the
     * input to be scanned
     */
    private int firstTokenIndex;
    /**
     * If true the tokens will be included in
     * the output.
     */
    private boolean includeTokensInOutput;
    /**
     *
     * @param input The input to scan.
     * @param includeTokensInOutput Will allow the splitting tokens to be added to the
     * final scan if this attribute is set to true.
     * @param splitterTokens An array of tokens..input as a
     * variable argument list... on which the input is to be split.
     */
    public CustomScannerOld( String input, boolean includeTokensInOutput, String... splitterTokens ) {
        this.input = new StringBuilder(input);
        this.splittingTokens = (List<String>) Arrays.asList(splitterTokens);
        this.includeTokensInOutput = includeTokensInOutput;
    }
    /**
     * A convenience constructor used when there exists
     * more than one array containing the tokenizer data.
     * @param input The input to scan.
     * @param includeTokensInOutput Will allow the splitting tokens to be added to the
     * final scan if this attribute is set to true.
     * @param splitterTokens1  An array of tokens..input as a
     * variable argument list... on which the input is to be split.
     * @param splitterTokens2  A second array of tokens..input as a
     * variable argument list... on which the input is to be split.
     *
     */
    public CustomScannerOld( String input, boolean includeTokensInOutput, String[] splitterTokens1, String... splitterTokens2 ) {
        this.input = new StringBuilder(input);
        splittingTokens.addAll(Arrays.asList(splitterTokens1));
        splittingTokens.addAll(Arrays.asList(splitterTokens2));
        this.includeTokensInOutput = includeTokensInOutput;
    }
    /**
     * A convenience constructor used when there exists
     * more than one array containing the tokenizer data.
     * @param input The input to scan.
     * @param includeTokensInOutput Will allow the splitting tokens to be added to the
     * final scan if this attribute is set to true.
     * @param splitterTokens An array of tokens on which the input is to be split.
     * @param splitterTokens1  A second array of tokens on which the input is to be split.
     * @param splitterTokens2  A second array of tokens..input as a
     * variable argument list... on which the input is to be split.
     *
     */
    public CustomScannerOld( String input, boolean includeTokensInOutput, String[] splitterTokens,String[] splitterTokens1, String... splitterTokens2 ) {
        this.input = new StringBuilder(input);

        splittingTokens.addAll(Arrays.asList(splitterTokens));
        splittingTokens.addAll(Arrays.asList(splitterTokens1));
        splittingTokens.addAll(Arrays.asList(splitterTokens2));
        this.includeTokensInOutput = includeTokensInOutput;
    }


    /**
     *
     * @return the length of the longest token
     * in the splittingTokens array.
     */
    private int getLongestSplitterTokenLength(){
        int max = splittingTokens.get(0).length();
        int size = splittingTokens.size();
        for(int i = 0; i < size; i++){
            int newmax = splittingTokens.get(i).length();
            if( max < newmax ){
                max = newmax;
            }
        }//end for
        return max;
    }
    /**
     *
     * @param testToken The string token to be tested.
     * @return true if the object is defined as one of
     * the elements in the splitting token array.
     */
    private boolean isSplittingToken( String testToken ){
        return splittingTokens.indexOf(testToken) != -1;
    }


    /**
     *
     * @return the first occurrence of an element of
     * the splittingToken array in the string to be split.
     */

    private String getFirstSplittingTokenInString(){
        String opString="";
        int longestSplittingToken = getLongestSplitterTokenLength();
        for(int i=0;i<input.length();i++){
            for(int j=longestSplittingToken;j>=0;j--){

                try{
                    if(i+j<=input.length() && isSplittingToken(input.substring(i,i+j))){
                        opString=input.substring(i,i+j);
                        setFirstTokenIndex(i);
                        return opString;
                    }//end if
                }//end try
                catch(IndexOutOfBoundsException ind){ind.printStackTrace();
                    continue;
                }//end catch
            }//end inner for
        }//end outer for

        return opString;
    }//end method getFirstOperatorInString

    /**
     *
     * @return an array containing any of the
     * elements in the splittingTokens array that
     * are also present in the input.
     */
    public  List<String> getFoundTokens(){
        List<String>split=new ArrayList<String>();

        while(input.length()>0){
            String op=getFirstSplittingTokenInString();
            if(!op.equals("")){
                split.add(op);
                input = new StringBuilder(input.substring( input.indexOf(op)+op.length()));

            }//end if
            else if(op.equals("")){
                break;
            }
        }//end loop
        return split;
    }//end method getOperatorStrings

    public void setFirstTokenIndex(int firstTokenIndex) {
        this.firstTokenIndex = firstTokenIndex;
    }

    public int getFirstTokenIndex() {
        return firstTokenIndex;
    }

    public void setIncludeTokensInOutput(boolean includeTokensInOutput) {
        this.includeTokensInOutput = includeTokensInOutput;
    }

    public boolean isIncludeTokensInOutput() {
        return includeTokensInOutput;
    }
    /**
     * Allows the object of this class to be reused over
     * for another input.
     * @param input sets the input to be scanned to this one.
     * If a new set of tokens should be used to split this string
     * the user should call method setSplittingTokens after calling this
     * method.
     */
    public void setInput(String input) {
        this.input = new StringBuilder(input);;
    }

    public String getInput() {
        return input.toString();
    }

    public void setSplittingTokens(List<String> splittingTokens) {
        this.splittingTokens = splittingTokens;
    }
    /**
     * Adds the values specified in the variable argument list
     * to its splitting tokens list.
     * @param splittingTokens A variable argument list containing
     * values to be added to the splitting tokens list of this scanner.
     */
    public void addSplittingTokens(String...splittingTokens){
        this.splittingTokens.addAll(Arrays.asList(splittingTokens));
    }
    /**
     * Adds the value specified as the method's argument
     * to its splitting tokens list.
     * @param splittingTokens A string containing the value
     * to be added to the splitting tokens list of this scanner.
     */
    public void addSplittingToken(String splittingTokens){
        this.splittingTokens.add(splittingTokens);
    }
    /**
     *
     * @param splittingTokens A variable argument list containing values that
     * are to be set as this scanner's tokens to be used for splitting the input.
     * All other splitting tokens specified before making a call to this method are discarded.
     */
    public void setSplittingTokens(String...splittingTokens) {
        this.splittingTokens = (List<String>) Arrays.asList(splittingTokens);
    }
    /**
     * Convenience method that takes two parameters, as specified below::
     * @param splittingTokens An array containing values that
     * are to be set as this scanner's tokens to be used for splitting the input.
     * @param splittingTokens1 A variable argument list containing more values to be added
     * to the to this scanner's splitting tokens.
     */
    public void setSplittingTokens(String[] splittingTokens,String...splittingTokens1) {
        this.splittingTokens = (List<String>) Arrays.asList(splittingTokens);
        this.splittingTokens.addAll(Arrays.asList(splittingTokens1));
    }



    public List<String> getSplittingTokens() {
        return splittingTokens;
    }



    /**
     *
     * @return this string split on the operators.
     */
    public List<String> scan(){

        List<String>scanner=new ArrayList<String>();
        List<String>filter=new ArrayList<String>();
        filter.add("");
        int opIndex=2;
        String op="<><./?>";
        int pass=0;
        if( includeTokensInOutput ){


            while(!op.equals("")){
                op = getFirstSplittingTokenInString();
                opIndex=getFirstTokenIndex();
                if(opIndex==0){

                    scanner.add(op);
                    this.input = new StringBuilder(input.substring(opIndex+op.length()));
                }
                else{
                    scanner.add(input.substring(0,opIndex));
                    scanner.add(op);
                    this.input = new StringBuilder(input.substring(opIndex+op.length()));
                }
                op = getFirstSplittingTokenInString();
                if(op.equals("")){
                    scanner.add(input.toString());
                }
                ++pass;

            }//end while


            scanner.removeAll(filter);
        }
        else{


            while(!op.equals("")){
                op = getFirstSplittingTokenInString();
                opIndex=getFirstTokenIndex();
                if(opIndex==0){
                    this.input = new StringBuilder(input.substring(opIndex+op.length()));
                }
                else{
                    scanner.add(input.substring(0,opIndex));
                    this.input = new StringBuilder(input.substring(opIndex+op.length()));
                }
                op = getFirstSplittingTokenInString();
                if(op.equals("")){
                    scanner.add(input.toString());
                }
                ++pass;

            }//end while


            scanner.removeAll(filter);
        }
        return scanner;
    }





    public static void main( String[] args ){
   /* String code = "function Matrix applyTouch( Integer a1, Matrix a2 )";
    CustomScanner customScanner = new CustomScanner(code, true,",","(",")"," ");
    System.out.println(customScanner.scan());
    */
        String code = "28+32+11-9E12+sin(3.2E9/cos(-3))-sinsinh(5)";

        CustomScannerOld customScanner = new CustomScannerOld(code, true,"sinh","+","-",")","(","sin","cos","/");
        System.out.println(customScanner.scan());

        customScanner.setInput("3sin2-8cos9/7+12");
        customScanner.setSplittingTokens("sin","cos","-","+","/");
        customScanner.setIncludeTokensInOutput(false);
        System.out.println(customScanner.scan());


    }





}