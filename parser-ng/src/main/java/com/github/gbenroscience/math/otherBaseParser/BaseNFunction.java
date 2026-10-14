/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.math.otherBaseParser;

import com.github.gbenroscience.parser.MathExpression;

/**
 * A {@link MathExpression} whose numbers are written, and whose answer is
 * printed, in an arbitrary number base from 2 to 36.
 * <p>
 * Digits are {@code 0-9} then {@code A-Z} (case-insensitive). The numbers in the
 * expression, including those in assignments such as {@code a=101;b=11;a*b}, are
 * read in the base given to the constructor, the arithmetic is done on the
 * values they stand for, and {@link #solve()} writes the result back in that
 * base. For example, in base 2, {@code 101+11} is {@code 5+3} and the answer is
 * {@code 1000.0}; in base 16, {@code 0FF+1} is {@code 255+1} and the answer is
 * {@code 100.0}.
 * <p>
 * How numbers are recognised (see
 * {@link BaseNumbers#toDecimalExpression(String, int)} for the full rules):
 * <ul>
 * <li>A number starts with a digit {@code 0-9}. Names ({@code x}, {@code sin},
 * {@code FF}) start with a letter, so a hexadecimal number that begins with a
 * letter needs a leading zero: {@code 0FF}.</li>
 * <li>Names can contain digits ({@code x2}); those digits are not converted.</li>
 * <li>Matrix sizes in {@code @(2,2)(...)} and indices in {@code x[2]} are always
 * decimal.</li>
 * <li>Above base 10, a number directly followed by a name is ambiguous
 * ({@code 2abs(x)}); write {@code 2*abs(x)}. Scientific notation is only
 * available in base 10.</li>
 * </ul>
 * A digit that does not exist in the base (for example {@code 2} in base 2)
 * makes the constructor throw a {@link NumberFormatException}. Base 10 behaves
 * exactly like {@link MathExpression}.
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class BaseNFunction extends MathExpression {

    /**
     * The base the numbers of the expression were written in.
     */
    private final int inputBase;

    /**
     * The base that results are printed in.
     */
    private int baseOfOperation;

    /**
     * The expression exactly as it was supplied, numbers still in the base of
     * operation.
     */
    private final String originalExpression;

    /**
     * @param input The function to be evaluated, with its numbers written in
     * {@code baseOfOperation}. The general format is that of
     * {@link MathExpression}: variable and function assignments followed by an
     * expression to evaluate, e.g. {@code a=101;b=11;a*b}.
     * @param baseOfOperation The number base that objects of this class will use
     * in computations, 2 to 36.
     * @throws IllegalArgumentException if the base is not in 2..36
     * @throws NumberFormatException if a number in the input uses a digit that
     * does not exist in the base, or is ambiguous
     */
    public BaseNFunction(String input, int baseOfOperation) {
        super(BaseNumbers.toDecimalExpression(input, BaseNumbers.checkBase(baseOfOperation)));
        this.originalExpression = input;
        this.inputBase = baseOfOperation;
        this.baseOfOperation = baseOfOperation;
    }

    /**
     * Changes the base that results are <b>printed</b> in. The numbers of the
     * expression were read in the base given to the constructor and stay that
     * way; use {@link #getInputBase()} to see it.
     *
     * @param baseOfOperation the base for results, 2 to 36
     * @throws IllegalArgumentException if the base is not in 2..36
     */
    public void setBaseOfOperation(int baseOfOperation) {
        this.baseOfOperation = BaseNumbers.checkBase(baseOfOperation);
    }

    /**
     * @return the base that results are printed in
     */
    public int getBaseOfOperation() {
        return baseOfOperation;
    }

    /**
     * @return the base the numbers of the expression were written in
     */
    public int getInputBase() {
        return inputBase;
    }

    /**
     * @return the expression as it was supplied, numbers still in the input
     * base. (Use {@link #getExpression()} for the decimal form that is
     * evaluated.)
     */
    public String getOriginalExpression() {
        return originalExpression;
    }

    /**
     * Evaluates the expression and writes the answer in the base of operation.
     * Numbers and lists of numbers are converted; text such as {@code NaN},
     * {@code Infinity} or an error message is returned as it is.
     *
     * @throws NumberFormatException if the expression is not valid
     */
    @Override
    public String solve() {
        if (!isCorrectFunction()) {
            throw new NumberFormatException("Invalid math expression!");
        }
        return BaseNumbers.convertResult(super.solve(), baseOfOperation);
    }

    /**
     * Like {@link #solve()}, for an expression with variables. The values in
     * {@code args} are plain numbers (doubles have no base), in the order the
     * variables appear; the result is written in the base of operation.
     */
    @Override
    public String solve(double... args) {
        if (!isCorrectFunction()) {
            throw new NumberFormatException("Invalid math expression!");
        }
        return BaseNumbers.convertResult(super.solve(args), baseOfOperation);
    }

    public static void main(String args[]) {
        // a=2, b=10 (binary 1010 -> decimal 10): prints the result of 2+10 in binary
        System.out.println(new BaseNFunction("a=10;b=1010;a+b", 2).solve());//1100.0
        System.out.println(new BaseNFunction("0FF+1", 16).solve());//100.0
    }

}//end class
