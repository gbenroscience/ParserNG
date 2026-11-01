package com.github.gbenroscience.gpu.llm.opencl;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.github.gbenroscience.gpu.llm.core.TinyModel;
import java.io.File;
import java.nio.file.Files;
import org.junit.jupiter.api.Test;

class OpenCLDiagnosticsTest {

    @Test
    void oneClickDiagnosticsRunsEveryStepAndWritesTheReportFile() throws Throwable {
        try {
            new GpuContext().close();
        } catch (Throwable t) {
            assumeTrue(false, "no usable OpenCL device: " + t);
        }
        TinyModel m = new TinyModel();
        m.dim = 128;
        m.hidden = 256;
        m.heads = 4;
        m.kvHeads = 2;
        m.build(51);
        File model = Files.createTempFile("diag", ".gguf").toFile();
        File out = Files.createTempFile("diag", ".txt").toFile();
        model.deleteOnExit();
        out.deleteOnExit();
        m.write(model);
        System.setProperty("llm.diagnostics.out", out.getAbsolutePath());
        try {
            OpenCLDiagnostics.main(new String[]{model.getAbsolutePath(), "--quick", "--max-new", "4"});
        } finally {
            System.clearProperty("llm.diagnostics.out");
            System.clearProperty("llm.bandwidth.mb");
        }
        String text = Files.readString(out.toPath());
        assertFalse(text.contains("STEP FAILED"), text);
        for (String needle : new String[]{"1/4 hardware and streaming bandwidth", "streaming read bandwidth", "2/4 device memory plan", "TOTAL predicted device memory",
            "3/4 GEMV sweep", "fit over the wg-per-row Q8_0 shapes", "4/4 full model", "load breakdown", "OpenCL event profile", "device memory by category",
            "logits readback", "summary (measured on this machine)"}) {
            assertTrue(text.contains(needle), "diagnostics output lacks '" + needle + "'");
        }
    }
}
