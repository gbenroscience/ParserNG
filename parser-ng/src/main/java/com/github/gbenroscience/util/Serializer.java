package com.github.gbenroscience.util;

import com.github.gbenroscience.interfaces.Savable;
import java.io.*;
import java.util.Base64;

/**
 * 1. Interface must extend Serializable for ObjectOutputStream to work.
 */
public class Serializer {

    /**
     * Converts an object to a Base64 encoded string.
     */
    public static final String serialize(Savable yourObject) {
        byte[] data = ser(yourObject);
        return (data != null) ? Base64.getEncoder().encodeToString(data) : null;
    }

    /**
     * Reconstructs an object from a Base64 encoded string.
     */
    public static final Savable deserialize(String encoded) {
        if (encoded == null || encoded.isEmpty()) {
            return null;
        }
        byte[] data = Base64.getDecoder().decode(encoded);
        return (Savable) deSer(data);
    }

    /**
     * Internal: Object to Byte Array
     */
    public static final byte[] ser(Serializable yourObject) {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream(); ObjectOutputStream out = new ObjectOutputStream(bos)) {
            out.writeObject(yourObject);
            out.flush();
            return bos.toByteArray();
        } catch (IOException e) {
            e.printStackTrace(); // Always print errors during debugging
            return null;
        }
    }

    /**
     * Internal: Byte Array to Object
     */
    private static final Object deSer(byte[] yourBytes) {
        if (yourBytes == null) {
            return null;
        }
        try (ByteArrayInputStream bis = new ByteArrayInputStream(yourBytes); ObjectInputStream in = new ObjectInputStream(bis)) {
            return in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Performs a 100% deep clone of any Serializable object (including 2D
     * arrays).
     *
     * @param <T> The type of the object
     * @param object The object to clone
     * @return A completely independent copy of the object
     */
    @SuppressWarnings("unchecked")
    public static <T extends Serializable> T deepClone(T object) {
        if (object == null) {
            return null;
        }

        // We use the existing ser/deSer logic
        // This creates a brand new object graph in memory
        return (T) deSer(ser(object));
    }

    /**
     * Test Class: Must be static so it doesn't try to serialize the
     * "Serializer" class along with it.
     */
    public static class Box implements Savable {

        private static final long serialVersionUID = 1L; // Ensures compatibility

        String name;
        int len, brd, hei;

        public Box(String name, int len, int brd, int hei) {
            this.name = name;
            this.len = len;
            this.brd = brd;
            this.hei = hei;
        }

        @Override
        public String toString() {
            return "Box{name='" + name + "', dimensions=[" + len + "," + brd + "," + hei + "]}";
        }
    }

    public static void main(String[] args) {
        // Test 1: Single Object
        Box bx = new Box("kolo-dollar", 12, 13, 5);
        String encoded = Serializer.serialize(bx);
        System.out.println("Encoded String: " + encoded);

        Box decodedBox = (Box) Serializer.deserialize(encoded);
        System.out.println("Reconstructed box: " + decodedBox);

        // Test 2: Complex 2D Array
        Box[][] grid = {
            {new Box("Top-Left", 1, 1, 1), new Box("Top-Right", 2, 2, 2)},
            {new Box("Bottom-Left", 3, 3, 3), new Box("Bottom-Right", 4, 4, 4)}
        };

        // Arrays are automatically Serializable if their components are!
        byte[] gridBytes = Serializer.ser(grid);
        Box[][] reconstructedGrid = (Box[][]) Serializer.deSer(gridBytes);

        System.out.println("2D Array Row 1, Col 1: " + reconstructedGrid[1][1]);
    }
}
