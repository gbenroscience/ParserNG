package com.github.gbenroscience.util;

import com.github.gbenroscience.parser.Bracket;
import java.io.File;

import com.github.gbenroscience.util.io.FunctionsBackup;
import com.github.gbenroscience.util.io.VariableBackup;
import com.github.gbenroscience.util.io.TextFileReader;
import com.github.gbenroscience.util.io.TextFileWriter;
import java.util.List;

public class Utils {

    /**
     * This file contains a record of all Variables
     */
    public static File VARIABLES;
    /**
     * This file contains a record of all Functions
     */
    public static File FUNCTIONS;
    /**
     * This file contains a record of all Settings
     */
    public static File SETTINGS;

    public static File APP_FOLDER;

    public static int IMAGE_ICON_SIZE = 300;

    public Utils() {

        String appName = "ParserNG";
        APP_FOLDER = new File(appName);

        VARIABLES = new File(APP_FOLDER.getAbsolutePath(), "variables.txt");
        FUNCTIONS = new File(APP_FOLDER.getAbsolutePath(), "functions.txt");
        SETTINGS = new File(APP_FOLDER.getAbsolutePath(), "settings.txt");
        try {
            if (!APP_FOLDER.exists()) {
                APP_FOLDER.mkdir();
            }
            if (!VARIABLES.exists()) {
                VARIABLES.createNewFile();
            }
            if (!FUNCTIONS.exists()) {
                FUNCTIONS.createNewFile();
            }
            if (!SETTINGS.exists()) {
                SETTINGS.createNewFile();
            }
        } catch (Exception e) {

        }

    }

    public static boolean loggingEnabled = false;

    public static void saveVariables() {
        synchronized (VariableManager.VARIABLES) {
            VariableBackup.writeMapItemsToFileLineByLine(VariableManager.VARIABLES, VARIABLES);
        }
    }

    public static void loadVariables() {
        VariableBackup.readFileLinesToMap(VariableManager.VARIABLES, VARIABLES);

    }

    public static void saveFunctions() {
        synchronized (FunctionManager.FUNCTIONS) {
            FunctionsBackup.writeMapItemsToFileLineByLine(FunctionManager.FUNCTIONS, FUNCTIONS);
        }
    }

    /**
     * loads the stored functions from persistent storage
     */
    public static void loadFunctions() {
        FunctionsBackup.readFileLinesToMap(FunctionManager.FUNCTIONS, FUNCTIONS);
    }

    /**
     * Saves the settings
     */
    public static void saveSettings(Settings settings) {
        String ser = settings.serialize();
        TextFileWriter.writeText(SETTINGS, ser);

    }

    public static Settings loadSettings() {

        String ser = new TextFileReader(SETTINGS).read();
        if (ser == null || ser.isEmpty()) {
            return null;
        }

        return Settings.parse(ser);
    }

    public static void saveAll(final Settings settings) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                FunctionManager.initializeFunctionVars();
                saveVariables();
                saveFunctions();
                saveSettings(settings);
            }
        }).start();

    }

    public static void loadAll() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                loadVariables();
                loadFunctions();
                loadSettings();
            }
        }).start();
    }

    public static void logError(String message) {
        if (loggingEnabled) {
            Log.e("Kalculitzer", message);
        }
    }

    public static void logDebug(String message) {
        if (loggingEnabled) {
            Log.d("Kalculitzer", message);
        }
    }

    public static void logInfo(String message) {
        if (loggingEnabled) {
            Log.i("Kalculitzer", message);
        }
    }

    public static boolean isAndroid() {
        String vendor = System.getProperty("java.vendor", "");
        String runtime = System.getProperty("java.runtime.name", "");
        return vendor.toLowerCase().contains("android")
                || runtime.toLowerCase().contains("android");
    }

    public static boolean isAndroidEmulator() {
        String runtime = System.getProperty("java.runtime.name", "");
        String vmName = System.getProperty("java.vm.name", "");

        // Check if we are on Android first
        if (runtime.toLowerCase().contains("android") || vmName.toLowerCase().contains("dalvik")) {
            // Properties often set in emulators
            String hardware = System.getProperty("ro.hardware", "");
            String kernel = System.getProperty("os.arch", "");

            // BlueStacks and other emulators often present specific hardware strings
            return hardware.contains("goldfish")
                    || hardware.contains("ranchu")
                    || hardware.contains("vbox86");
        }
        return false;
    }

    public static boolean isPerfectSquare(int num) {
        if (num < 0) {
            return false;
        }
        int sqrt = (int) Math.sqrt(num);
        return sqrt * sqrt == num;
    }

    public static void unwrapBracket(List<String> data) {
        if (data.isEmpty()) {
            return;
        }
        if (data.get(0).equals("(")) {
            int closeIdx = Bracket.getComplementIndex(true, 0, data);
            if (closeIdx == data.size() - 1) {
                data.remove(closeIdx);
                data.remove(0);
            }
        }
    }

    public static String unwrapBracket(String data) {
        if (data.isEmpty()) {
            return data;
        }
        if (data.charAt(0) == '(') {
            int closeIdx = Bracket.getComplementIndex(true, 0, data);
            if (closeIdx == data.length() - 1) {
                data = data.substring(1, closeIdx);
            }
        }
        return data;
    }

    public static void main(String[] args) {
        System.out.println("isAndroid: " + isAndroid());
    }
}
