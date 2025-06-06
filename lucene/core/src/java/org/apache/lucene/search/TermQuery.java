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
package org.apache.lucene.search;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Objects;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermState;
import org.apache.lucene.index.TermStates;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.util.IOSupplier;

/**
 * A Query that matches documents containing a term. This may be combined with other terms with a
 * {@link BooleanQuery}.
 */
public class TermQuery extends Query {

  private final Term term;
  private final TermStates perReaderTermState;

  final class TermWeight extends Weight {
    private final Similarity similarity;
    private final Similarity.SimScorer simScorer;
    private final TermStates termStates;
    private final ScoreMode scoreMode;

    public TermWeight(
        IndexSearcher searcher, ScoreMode scoreMode, float boost, TermStates termStates)
        throws IOException {
      super(TermQuery.this);
      if (scoreMode.needsScores() && termStates == null) {
        throw new IllegalStateException("termStates are required when scores are needed");
      }
      this.scoreMode = scoreMode;
      this.termStates = termStates;
      this.similarity = searcher.getSimilarity();

      final CollectionStatistics collectionStats;
      final TermStatistics termStats;
      if (scoreMode.needsScores()) {
        collectionStats = searcher.collectionStatistics(term.field());
        termStats =
            termStates.docFreq() > 0
                ? searcher.termStatistics(term, termStates.docFreq(), termStates.totalTermFreq())
                : null;
      } else {
        // we do not need the actual stats, use fake stats with docFreq=maxDoc=ttf=1
        collectionStats = new CollectionStatistics(term.field(), 1, 1, 1, 1);
        termStats = new TermStatistics(term.bytes(), 1, 1);
      }

      if (termStats == null) {
        this.simScorer = null; // term doesn't exist in any segment, we won't use similarity at all
      } else {
        // Assigning a dummy simScorer in case score is not needed to avoid unnecessary float[]
        // allocations in case default BM25Scorer is used.
        // See: https://github.com/apache/lucene/issues/12297
        if (scoreMode.needsScores()) {
          this.simScorer = similarity.scorer(boost, collectionStats, termStats);
        } else {
          // Assigning a dummy scorer as this is not expected to be called since scores are not
          // needed.
          this.simScorer =
              new Similarity.SimScorer() {
                @Override
                public float score(float freq, long norm) {
                  return 0f;
                }
              };
        }
      }
    }

    @Override
    public Matches matches(LeafReaderContext context, int doc) throws IOException {
      TermsEnum te = getTermsEnum(context);
      if (te == null) {
        return null;
      }
      return MatchesUtils.forField(
          term.field(),
          () -> {
            PostingsEnum pe = te.postings(null, PostingsEnum.OFFSETS);
            if (pe.advance(doc) != doc) {
              return null;
            }
            return new TermMatchesIterator(getQuery(), pe);
          });
    }

    @Override
    public String toString() {
      return "weight(" + TermQuery.this + ")";
    }

    @Override
    public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
      assert termStates == null || termStates.wasBuiltFor(ReaderUtil.getTopLevelContext(context))
          : "The top-reader used to create Weight is not the same as the current reader's top-reader ("
              + ReaderUtil.getTopLevelContext(context);

      final IOSupplier<TermState> stateSupplier = termStates.get(context);
      if (stateSupplier == null) {
        return null;
      }

      return new ScorerSupplier() {

        private TermsEnum termsEnum;
        private boolean topLevelScoringClause = false;

        private TermsEnum getTermsEnum() throws IOException {
          if (termsEnum == null) {
            TermState state = stateSupplier.get();
            if (state == null) {
              return null;
            }
            termsEnum = context.reader().terms(term.field()).iterator();
            termsEnum.seekExact(term.bytes(), state);
          }
          return termsEnum;
        }

        @Override
        public Scorer get(long leadCost) throws IOException {
          TermsEnum termsEnum = getTermsEnum();
          if (termsEnum == null) {
            return new ConstantScoreScorer(0f, scoreMode, DocIdSetIterator.empty());
          }

          NumericDocValues norms = null;
          if (scoreMode.needsScores()) {
            norms = context.reader().getNormValues(term.field());
          }

          if (scoreMode == ScoreMode.TOP_SCORES) {
            return new TermScorer(
                termsEnum.impacts(PostingsEnum.FREQS), simScorer, norms, topLevelScoringClause);
          } else {
            int flags = scoreMode.needsScores() ? PostingsEnum.FREQS : PostingsEnum.NONE;
            return new TermScorer(termsEnum.postings(null, flags), simScorer, norms);
          }
        }

        @Override
        public BulkScorer bulkScorer() throws IOException {
          if (scoreMode.needsScores() == false) {
            DocIdSetIterator iterator = get(Long.MAX_VALUE).iterator();
            int maxDoc = context.reader().maxDoc();
            return ConstantScoreScorerSupplier.fromIterator(iterator, 0f, scoreMode, maxDoc)
                .bulkScorer();
          }
          return new BatchScoreBulkScorer(get(Long.MAX_VALUE));
        }

        @Override
        public long cost() {
          try {
            TermsEnum te = getTermsEnum();
            return te == null ? 0 : te.docFreq();
          } catch (IOException e) {
            throw new UncheckedIOException(e);
          }
        }

        @Override
        public void setTopLevelScoringClause() {
          topLevelScoringClause = true;
        }
      };
    }

    @Override
    public boolean isCacheable(LeafReaderContext ctx) {
      return true;
    }

    /**
     * Returns a {@link TermsEnum} positioned at this weights Term or null if the term does not
     * exist in the given context
     */
    private TermsEnum getTermsEnum(LeafReaderContext context) throws IOException {
      assert termStates != null;
      assert termStates.wasBuiltFor(ReaderUtil.getTopLevelContext(context))
          : "The top-reader used to create Weight is not the same as the current reader's top-reader ("
              + ReaderUtil.getTopLevelContext(context);
      final IOSupplier<TermState> supplier = termStates.get(context);
      final TermState state = supplier == null ? null : supplier.get();
      if (state == null) { // term is not present in that reader
        assert termNotInReader(context.reader(), term)
            : "no termstate found but term exists in reader term=" + term;
        return null;
      }
      final TermsEnum termsEnum = context.reader().terms(term.field()).iterator();
      termsEnum.seekExact(term.bytes(), state);
      return termsEnum;
    }

    private boolean termNotInReader(LeafReader reader, Term term) throws IOException {
      // only called from assert
      // System.out.println("TQ.termNotInReader reader=" + reader + " term=" +
      // field + ":" + bytes.utf8ToString());
      return reader.docFreq(term) == 0;
    }

    @Override
    public Explanation explain(LeafReaderContext context, int doc) throws IOException {
      Scorer scorer = scorer(context);
      if (scorer != null) {
        int newDoc = scorer.iterator().advance(doc);
        if (newDoc == doc) {
          float freq = ((TermScorer) scorer).freq();
          NumericDocValues norms = context.reader().getNormValues(term.field());
          long norm = 1L;
          if (norms != null && norms.advanceExact(doc)) {
            norm = norms.longValue();
          }
          Explanation freqExplanation =
              Explanation.match(freq, "freq, occurrences of term within document");
          Explanation scoreExplanation = simScorer.explain(freqExplanation, norm);
          return Explanation.match(
              scoreExplanation.getValue(),
              "weight("
                  + getQuery()
                  + " in "
                  + doc
                  + ") ["
                  + similarity.getClass().getSimpleName()
                  + "], result of:",
              scoreExplanation);
        }
      }
      return Explanation.noMatch("no matching term");
    }

    @Override
    public int count(LeafReaderContext context) throws IOException {
      if (context.reader().hasDeletions() == false) {
        TermsEnum termsEnum = getTermsEnum(context);
        // termsEnum is not null if term state is available
        if (termsEnum != null) {
          return termsEnum.docFreq();
        } else {
          // the term cannot be found in the dictionary so the count is 0
          return 0;
        }
      } else {
        return super.count(context);
      }
    }
  }

  /** Constructs a query for the term <code>t</code>. */
  public TermQuery(Term t) {
    term = Objects.requireNonNull(t);
    perReaderTermState = null;
  }

  /**
   * Expert: constructs a TermQuery that will use the provided docFreq instead of looking up the
   * docFreq against the searcher.
   */
  public TermQuery(Term t, TermStates states) {
    assert states != null;
    term = Objects.requireNonNull(t);
    perReaderTermState = Objects.requireNonNull(states);
  }

  /** Returns the term of this query. */
  public Term getTerm() {
    return term;
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost)
      throws IOException {
    final IndexReaderContext context = searcher.getTopReaderContext();
    final TermStates termState;
    if (perReaderTermState == null || perReaderTermState.wasBuiltFor(context) == false) {
      termState = TermStates.build(searcher, term, scoreMode.needsScores());
    } else {
      // PRTS was pre-build for this IS
      termState = this.perReaderTermState;
    }

    return new TermWeight(searcher, scoreMode, boost, termState);
  }

  @Override
  public void visit(QueryVisitor visitor) {
    if (visitor.acceptField(term.field())) {
      visitor.consumeTerms(this, term);
    }
  }

  /** Prints a user-readable version of this query. */
  @Override
  public String toString(String field) {
    StringBuilder buffer = new StringBuilder();
    if (!term.field().equals(field)) {
      buffer.append(term.field());
      buffer.append(":");
    }
    buffer.append(term.text());
    return buffer.toString();
  }

  /**
   * Returns the {@link TermStates} passed to the constructor, or null if it was not passed.
   *
   * @lucene.experimental
   */
  public TermStates getTermStates() {
    return perReaderTermState;
  }

  /** Returns true iff <code>other</code> is equal to <code>this</code>. */
  @Override
  public boolean equals(Object other) {
    return sameClassAs(other) && term.equals(((TermQuery) other).term);
  }

  @Override
  public int hashCode() {
    return classHash() ^ term.hashCode();
  }
}
