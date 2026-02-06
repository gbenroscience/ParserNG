package com.github.gbenroscience.parser.methods.ext;

import java.io.File;
import java.nio.file.Files;
import java.util.List;
import java.util.stream.Collectors;

import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.methods.BasicNumericalMethod;
import com.github.gbenroscience.parser.methods.Help;

public class Echo implements BasicNumericalMethod {


    @Override
    public String solve(List<String> tokens) {
        return tokens.stream().collect(Collectors.joining(" "));
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), " simply reprints its space dleimited arguments. Result is final, and no other operations possible");
    }

    @Override
    public String getName() {
        return "echo";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }

    public static class EchoN implements BasicNumericalMethod {

        @Override
        public String solve(List<String> tokens) {
            return tokens.stream().collect(Collectors.joining(System.lineSeparator()));
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), " simply reprints its new-line delimited arguments. Result is final, and no other operations possible");
        }

        @Override
        public String getName() {
            return "echon";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }

    public static class EchoNI implements BasicNumericalMethod {

        @Override
        public String solve(List<String> tokens) {
            StringBuilder stringBuilder = new StringBuilder();
            for (int i = 0; i < tokens.size(); i++) {
                stringBuilder.append(i).append(":").append(tokens.get(i)).append(System.lineSeparator());
            }
            return stringBuilder.toString();
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), " simply reprints its new-line delimited arguments with its Index. Result is final, and no other operations possible");
        }

        @Override
        public String getName() {
            return "echoni";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }

    public static class EchoI implements BasicNumericalMethod {

        @Override
        public String solve(List<String> tokens) {
            StringBuilder stringBuilder = new StringBuilder();
            for (int i = 0; i < tokens.size(); i++) {
                stringBuilder.append(i).append(":").append(tokens.get(i)).append(" ");
            }
            return stringBuilder.toString();
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), " simply reprints its space delimited arguments with its Index. Result is final, and no other operations possible");
        }

        @Override
        public String getName() {
            return "echoi";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }
}
