/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.benchmark.jmh;

import java.io.IOException;
import java.nio.file.Files;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 5)
@Fork(
    value = 0,
    jvmArgsPrepend = {"--add-modules=jdk.unsupported", "--add-modules=jdk.incubator.vector"})
public class NeighborListsBenchMark {
  Directory dir;
  HnswGraph g;
  IndexReader reader;

  @Param({"true", "false"})
  public boolean baseline;

  @Setup(Level.Trial)
  public void init() throws Exception {
    dir = new ByteBuffersDirectory();
    Directory dir = new MMapDirectory(Files.createTempDirectory("neighbor"));
    writeIndex(dir);
    reader = DirectoryReader.open(dir);
    LeafReader leaf = reader.leaves().get(0).reader();
    g =
        ((Lucene99HnswVectorsReader)
                ((PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader) leaf).getVectorReader())
                    .getFieldReader("field"))
            .getGraph("field");
  }

  @TearDown
  public void tearDown() throws IOException {
    dir.close();
    reader.close();
  }

  @Benchmark
  public void doSeek() throws IOException {
    for (int level = 0; level < g.numLevels(); level++) {
      HnswGraph.NodesIterator nodesOnLevel = g.getNodesOnLevel(level);
      while (nodesOnLevel.hasNext()) {
        int node = nodesOnLevel.nextInt();
        g.seek(level, node);
      }
    }
  }

  void writeIndex(Directory dir) throws IOException {
    Random r = new Random(0);
    int dim = 100;
    int numDoc = 500;
    int maxConn = 16;
    int beamWidth = 100;

    byte[] vectorValue = new byte[dim];

    IndexWriterConfig iwc =
        new IndexWriterConfig()
            .setOpenMode(IndexWriterConfig.OpenMode.CREATE)
            .setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    Lucene99HnswVectorsFormat fmt = new Lucene99HnswVectorsFormat(maxConn, beamWidth);
    Lucene99HnswVectorsFormat.baseline = this.baseline;
    iwc.setCodec(
        new FilterCodec(iwc.getCodec().getName(), iwc.getCodec()) {
          @Override
          public KnnVectorsFormat knnVectorsFormat() {
            return new PerFieldKnnVectorsFormat() {
              @Override
              public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return fmt;
              }
            };
          }
        });
    IndexWriter w = new IndexWriter(dir, iwc);
    for (int i = 0; i < numDoc; i++) {
      Document doc = new Document();
      r.nextBytes(vectorValue);
      doc.add(new KnnByteVectorField("field", vectorValue, VectorSimilarityFunction.EUCLIDEAN));
      doc.add(new StoredField("id", i));
      w.addDocument(doc);
    }
    w.commit();
    w.close();
  }
}
