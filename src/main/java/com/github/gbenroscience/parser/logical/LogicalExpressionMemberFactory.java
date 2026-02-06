package com.github.gbenroscience.parser.logical;

public interface LogicalExpressionMemberFactory {

    LogicalExpressionMember createLogicalExpressionMember(String expression, ExpressionLogger log);

    interface LogicalExpressionMember {

        /**
         * Help for this parser
         */
        String getHelp();

        /**
         * The method should understand true/false strings, if it is supposed to be used in more complicated expressions.
         *
         * @return evaluated expression, usually parsed in constructor
         */
        boolean evaluate();

        /**
         * ParserNG have a habit, that expression is parsed in constructor, and later evaluated in methood.
         * So this methid is takin parameter, of future expression, created over dummy example, so we know,
         * whether it will be viable for future constructor.
         *
         * @param futureExpression future expression to be passed to constructor
         * @return whether the expression is most likely targeted for this parser
         */
        boolean isLogicalExpressionMember(String futureExpression);


    }
}
