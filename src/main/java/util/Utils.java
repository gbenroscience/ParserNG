package util;

import java.io.File;

import util.io.FunctionsBackup;
import util.io.VariableBackup;
import util.io.TextFileReader;
import util.io.TextFileWriter;

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

}
