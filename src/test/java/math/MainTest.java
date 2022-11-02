package math;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class MainTest {

    @Test
    void help() {
        Main.help();
    }

    @Test
    void examples() {
        Main.examples();
    }

    @Test
    void joinArgs() {
        String r;
        r = Main.joinArgs(Arrays.asList("1", "+", "1"), true);
        Assertions.assertEquals("1 + 1 ", r);
        r = Main.joinArgs(Arrays.asList("1", "+", "1"), false);
        Assertions.assertEquals("1 + 1 ", r);
        r = Main.joinArgs(Arrays.asList("1\n", "+\n", "1\n"), true);
        Assertions.assertEquals("1  +  1  ", r);
        r = Main.joinArgs(Arrays.asList("1\n", "+\n", "1\n"), false);
        Assertions.assertEquals("1\n"
                + " +\n"
                + " 1\n"
                + " ", r);
    }
}