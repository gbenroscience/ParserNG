package parser;


import java.util.InputMismatchException;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;


public class BigMathExpressionTest {

    @Test
    void contrabandTestShouldPassSimpleExpression() {
        BigMathExpression me = null;
        try{
            me = new BigMathExpression("x=2;5*x^2+x+2*x;");
            Assertions.assertEquals(true, me.isCorrectFunction());
        }catch(InputMismatchException e){
            Assertions.assertEquals(true, false);
        }

        
    }

    @Test
    void contrabandTestShouldPassFunctionWithSimpleExpression() {
        BigMathExpression me = null;
        try{
            me = new BigMathExpression("x=2;h(x)=5*x^2+x+2*x;h(3);");
            Assertions.assertEquals(true, me.isCorrectFunction());
        }catch(InputMismatchException e){
            Assertions.assertEquals(true, false);
        }

        
    }

    @Test
    void contrabandTestShouldFailInbuiltFunctionFound() {
        BigMathExpression me = null;
        try{
            me = new BigMathExpression("x=2;5*x^2+x+ln(x);");
            Assertions.assertEquals(true, me.isCorrectFunction());
        }catch(InputMismatchException e){
            Assertions.assertEquals(false, false);
        }

       
    }

    @Test
    void contrabandTestShouldFailUserDefinedFunctionHasInbuiltFunction() {
        BigMathExpression me = null;
        try{
            me = new BigMathExpression("x=2;h(x)=5*x^2+x+2*x+cosh(x);h(4);");
            Assertions.assertEquals(false, me.isCorrectFunction());
        }catch(InputMismatchException e){
            Assertions.assertEquals(false, false);
        }
    }
    
}
