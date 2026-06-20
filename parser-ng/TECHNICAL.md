# **ParserNG: Performance & Memory Manifesto**

ParserNG is a high-performance mathematical expression engine for Java, designed for systems where **nanoseconds matter** and **Garbage Collection (GC) is unacceptable.**

In **version 2.0.0**, ParserNG goes beyond turbo-execution with MethodHandles, and explores the use of

1. The Vector API to vectorize code execution
2. Mechanically sympathetic code to trigger auto-vectorization by the underlying hardware. This leads to equal or nigh-equal
performance with the performance gotten from the Vector API, alongside zero allocation


For benchmarks showing ParserNG duking it out in the ring with other parsers, 
[check ParserNg-wars]( https://github.com/gbenroscience/ParserNg-wars )

## **Summary**
If you are building high-frequency trading platforms, real-time simulation software, or cloud-native microservices with tight memory budgets, **ParserNG** provides the deterministic speed you need without the Garbage Collection or Metaspace risks of traditional tools.

---
 