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
package org.apache.lucene.queries.function.valuesource;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.queries.function.FunctionValues;
import org.apache.lucene.queries.function.ValueSource;
import org.apache.lucene.search.DocIdSetIterator;

/**
 * An implementation for retrieving {@link FunctionValues} instances for byte knn vectors fields.
 */
public class ByteKnnVectorFieldSource extends ValueSource {
  private final String fieldName;

  public ByteKnnVectorFieldSource(String fieldName) {
    this.fieldName = fieldName;
  }

  @Override
  public FunctionValues getValues(Map<Object, Object> context, LeafReaderContext readerContext)
      throws IOException {

    final LeafReader reader = readerContext.reader();
    final ByteVectorValues vectorValues = reader.getByteVectorValues(fieldName);

    if (vectorValues == null) {
      VectorFieldFunction.checkField(reader, fieldName, VectorEncoding.BYTE);

      return new VectorFieldFunction(this) {
        private final DocIdSetIterator empty = DocIdSetIterator.empty();

        @Override
        public byte[] byteVectorVal(int doc) throws IOException {
          return null;
        }

        @Override
        protected DocIdSetIterator getVectorIterator() {
          return empty;
        }
      };
    }

    return new VectorFieldFunction(this) {
      KnnVectorValues.DocIndexIterator iterator = vectorValues.iterator();

      @Override
      public byte[] byteVectorVal(int doc) throws IOException {
        if (exists(doc)) {
          return vectorValues.vectorValue(iterator.index());
        } else {
          return null;
        }
      }

      @Override
      protected DocIdSetIterator getVectorIterator() {
        return iterator;
      }
    };
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    ByteKnnVectorFieldSource other = (ByteKnnVectorFieldSource) o;
    return Objects.equals(fieldName, other.fieldName);
  }

  @Override
  public int hashCode() {
    return Objects.hash(getClass().hashCode(), fieldName);
  }

  @Override
  public String description() {
    return "ByteKnnVectorFieldSource(" + fieldName + ")";
  }
}
