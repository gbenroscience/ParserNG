package parser.methods.ext;

import logic.DRG_MODE;
import math.Maths;
import parser.MathExpression;
import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Declarations;
import parser.methods.Help;

import java.util.List;

/**
 * todo, extract trigonometric ancestor for all methods dealing with setRadDegGrad, once more is done
 */
public class Sin implements BasicNumericalMethod {

    private DRG_MODE DRG = null;

    @Override
    /**
     * Warning! This method is not synchronised.
     * and the custom methods in parserng recycle same object
     * So if you are calling new MathExpression("sin(x)")); me.setRadDegGrad(differentValues); me.solve()
     * it can go wild. Maybe one day we will switch to new object per call
     */
    public void setRadDegGrad(DRG_MODE deg) {
        this.DRG=deg;
    }

    @Override
    public String solve(List<String> tokens) {
        String token;
        if (Utils.checkOnlyNumbers(tokens)) {
            if (tokens.size() != 1) {
                throw new RuntimeException("Sinus takes exactly one argument. Was " + tokens.size() + " (" + tokens.toString() + ")");
            }
            token = tokens.get(0);
        } else {
            token = new MathExpression(Utils.connectTokens(tokens)).solve();
        }
        if (DRG == null) {
            DRG = Declarations.degGradRadFromVariable();
        }
        if (DRG == DRG_MODE.DEG) {
            return String.valueOf(Maths.sinDegToRad(Double.valueOf(token)));
        } else if (DRG == DRG_MODE.RAD) {
            return String.valueOf(Math.sin(Double.valueOf(token)));
        } else if (DRG == DRG_MODE.GRAD) {
            return String.valueOf(Maths.sinGradToRad(Double.valueOf(token)));
        } else {
            throw new RuntimeException("Unknown DRG/RAD/GRAD mode - " + DRG);
        }
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), "us function. By default it uses Radians. Use variable or property of" + DRG_MODE.DEG_MODE_VARIABLE + "=DEG/RAD/GRAD variable to set it up./");
    }

    @Override
    public String getName() {
        return "sin";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
