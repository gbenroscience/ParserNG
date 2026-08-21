package com.github.gbenroscience.arrow.tools.box1;

/**
 * Thrown when a column ParserNG needs to read is not a {@code
 * org.apache.arrow.vector.Float8Vector} and isn't one of the numeric
 * types {@link VectorCoercion} knows how to convert either ({@code
 * IntVector}, {@code BigIntVector}, {@code Float4Vector} are all bound
 * transparently — see {@link ArrowBulkEvaluator}'s class javadoc).
 * Typically this means a non-numeric column ({@code VarCharVector},
 * {@code BitVector}, ...) or a numeric type {@link VectorCoercion}
 * hasn't been taught yet (e.g. {@code DecimalVector}).
 */
public class UnsupportedVectorTypeException extends RuntimeException {

    public UnsupportedVectorTypeException(String columnName, Class<?> actualType) {
        super("Column '" + columnName + "' is a " + actualType.getSimpleName()
                + ", which ArrowBulkEvaluator has no conversion path for. It binds "
                + "org.apache.arrow.vector.Float8Vector directly (zero-copy) and auto-coerces "
                + "IntVector, BigIntVector, and Float4Vector via VectorCoercion (a real copy) — "
                + "cast this column to one of those upstream in your Arrow pipeline, or add a case "
                + "for it in VectorCoercion if it's numeric.");
    }
}