package parser.logical;

public interface ExpressionLogger {

    public static final class InheritingExpressionLogger implements  ExpressionLogger {

        private final ExpressionLogger parent;

        public InheritingExpressionLogger(ExpressionLogger parent) {
            this.parent = parent;
        }

        @Override
        public void log(String s) {
            parent.log("  " + s);
        }
    }
    public static ExpressionLogger DEV_NULL = new ExpressionLogger() {
        @Override
        public void log(String s) {
        }
    };

    public void log(String s);
}
