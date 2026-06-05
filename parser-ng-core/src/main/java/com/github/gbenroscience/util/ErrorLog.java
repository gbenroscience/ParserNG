package com.github.gbenroscience.util;

import com.github.gbenroscience.interfaces.Savable;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringWriter;

/**
 * A utility class that tracks errors and warnings during parsing operations.
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class ErrorLog implements Savable {

    private static final long serialVersionUID = 1L;

    // Encapsulated state
    private final StringBuilder builder = new StringBuilder();

    // Enum to handle both Errors and Warnings
    public enum Level {
        ERROR, WARNING, INFO
    }

    /**
     * Logs a message with a specific severity level.
     *
     * @param level
     * @param message
     */
    public void log(Level level, String message) {
        builder.append(level.name()).append(" - ").append(message).append("\n");
    }

    /**
     *
     * @param message
     */
    public void info(String message) {
        builder.append(Level.INFO).append(" - ").append(message).append("\n");
    }

    /**
     * Convenience method for logging errors.
     *
     * @param message
     */
    public void error(String message) {
        log(Level.ERROR, message);
    }

    /**
     * Convenience method for logging warnings.
     *
     * @param message
     */
    public void warning(String message) {
        log(Level.WARNING, message);
    }

    /**
     * Logs an exception using a clean, standard stack trace format.
     *
     * @param t
     */
    public void error(Throwable t) {
        builder.append(Level.ERROR.name()).append(" - ")
                .append("Exception: ").append(t.getMessage()).append("\n");

        // Use standard Java stack trace formatting (much easier to read)
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        t.printStackTrace(pw);
        builder.append(sw.toString());
    }

    public String getLogs() {
        return builder.toString();
    }

    /**
     * Prints the contents of this log to System.print without clearing the log.
     */
    public void print() {
        print(false);
    }

    /**
     * Prints the contents of this log to System.print.
     *
     * @param flush If true, clears the log memory after printing.
     */
    public void print(boolean flush) {
        print(System.out, flush);
    }

    /**
     * Prints the contents of this log to a specified PrintStream.
     *
     * @param out The PrintStream to output to.
     * @param flush If true, clears the log memory after printing.
     */
    public void print(PrintStream out, boolean flush) {
        // Prevent printing empty lines if there's nothing to log
        if (builder.length() > 0) {
            out.print(builder.toString());
            if (flush) {
                clearLog();
            }
        }
    }

    public void clearLog() {
        // More efficient way to clear a StringBuilder
        builder.setLength(0);
    }

    public void copyTo(ErrorLog log) {
        if (log != null) {
            log.builder.append(this.builder);
        }
    }

    public void copyFrom(ErrorLog log) {
        if (log != null) {
            this.builder.append(log.builder);
        }
    }

    public static void main(String[] args) {
        ErrorLog logger = new ErrorLog();

        try {
            // Evaluates as integer division first, throws ArithmeticException
            int a = 1 / 0;
            System.out.println("a = " + a);
        } catch (ArithmeticException e) {
            logger.error(e);
            logger.print(true); // Print and flush
        }

        ErrorLog log = new ErrorLog();
        log.error("Parser- Syntax Error");
        log.warning("Unrecognized variable 'x', assuming 0"); // Now supports warnings!
        log.error("Parser- Evaluation Error");
        log.print();
    }
}
