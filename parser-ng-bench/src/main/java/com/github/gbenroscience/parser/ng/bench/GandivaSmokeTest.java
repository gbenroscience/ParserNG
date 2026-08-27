package com.github.gbenroscience.parser.ng.bench;
 
import org.apache.arrow.gandiva.evaluator.Projector;

public class GandivaSmokeTest {

    public static void main(String[] args) {
        System.out.println(Projector.class);
        System.out.println("Gandiva Java classes loaded.");
    }
}