package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ast.AndExpr;
import com.github.gbenroscience.sqlv1.ast.BetweenExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.CompOp;
import com.github.gbenroscience.sqlv1.ast.ComparisonExpr;
import com.github.gbenroscience.sqlv1.ast.InExpr;
import com.github.gbenroscience.sqlv1.ast.IsNullExpr;
import com.github.gbenroscience.sqlv1.ast.NotExpr;
import com.github.gbenroscience.sqlv1.ast.OrExpr;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Evaluates a {@code HAVING} over finished groups: comparisons, {@code BETWEEN}, {@code IN}, {@code IS NULL},
 * {@code AND}, {@code OR} and {@code NOT}, with SQL's three-valued logic (a null operand makes a comparison
 * unknown and only {@code TRUE} keeps a group). Operands are supplied by a resolver that maps the operand text to a
 * number per group; it throws for text it cannot resolve.
 */
final class HavingEval {

    private HavingEval() { }

    /** A numeric operand: the value for group {@code g}, or null. */
    interface Num {
        Double eval(int g);
    }

    interface Node {
        Boolean eval(int g);
    }

    static Node compile(BoolExpr e, Function<String, Num> operand) {
        return switch (e) {
            case AndExpr(var l, var r) -> {
                Node a = compile(l, operand), b = compile(r, operand);
                yield g -> {
                    Boolean x = a.eval(g), y = b.eval(g);
                    if (Boolean.FALSE.equals(x) || Boolean.FALSE.equals(y)) return false;
                    return x == null || y == null ? null : Boolean.TRUE;
                };
            }
            case OrExpr(var l, var r) -> {
                Node a = compile(l, operand), b = compile(r, operand);
                yield g -> {
                    Boolean x = a.eval(g), y = b.eval(g);
                    if (Boolean.TRUE.equals(x) || Boolean.TRUE.equals(y)) return true;
                    return x == null || y == null ? null : Boolean.FALSE;
                };
            }
            case NotExpr(var inner) -> {
                Node a = compile(inner, operand);
                yield g -> {
                    Boolean x = a.eval(g);
                    return x == null ? null : !x;
                };
            }
            case ComparisonExpr(var l, var op, var r) -> {
                Num a = operand.apply(l), b = operand.apply(r);
                yield g -> {
                    Double x = a.eval(g), y = b.eval(g);
                    return x == null || y == null ? null : cmp(op, x, y);
                };
            }
            case BetweenExpr(var t, var lo, var hi, var neg) -> {
                Num a = operand.apply(t), l = operand.apply(lo), h = operand.apply(hi);
                yield g -> {
                    Double x = a.eval(g), lv = l.eval(g), hv = h.eval(g);
                    if (x == null || lv == null || hv == null) return null;
                    boolean in = x >= lv && x <= hv;
                    return neg ? !in : in;
                };
            }
            case InExpr(var t, var vals, var neg) -> {
                Num a = operand.apply(t);
                List<Num> vs = new ArrayList<>();
                for (String v : vals) vs.add(operand.apply(v));
                yield g -> {
                    Double x = a.eval(g);
                    if (x == null) return null;
                    boolean hit = false, unknown = false;
                    for (Num v : vs) {
                        Double y = v.eval(g);
                        if (y == null) unknown = true;
                        else if (y.doubleValue() == x.doubleValue()) hit = true;
                    }
                    if (hit) return !neg;
                    return unknown ? null : (Boolean) neg;
                };
            }
            case IsNullExpr(var t, var neg) -> {
                Num a = operand.apply(t);
                yield g -> (a.eval(g) == null) != neg;
            }
            default -> throw new ParquetSqlException("HAVING supports comparisons, BETWEEN, IN, IS NULL, AND, OR and NOT over aggregates and numbers, not " + e);
        };
    }

    private static boolean cmp(CompOp op, double x, double y) {
        return switch (op) {
            case EQ -> x == y;
            case NEQ -> x != y;
            case LT -> x < y;
            case LE -> x <= y;
            case GT -> x > y;
            case GE -> x >= y;
        };
    }
}
