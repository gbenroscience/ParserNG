package parser.methods;

import logic.DRG_MODE;

import java.util.List;

public interface BasicNumericalMethod {

    String solve(List<String> tokens);

    String getHelp();

    String getName();

    String getType();

    default void setRadDegGrad(DRG_MODE deg) {
        //no op for most
    }

}
