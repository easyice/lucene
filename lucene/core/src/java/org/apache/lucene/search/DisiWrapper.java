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

import java.util.Objects;
import org.apache.lucene.index.PostingsEnum;

/**
 * Wrapper used in {@link DisiPriorityQueue}.
 *
 * @lucene.internal
 */
public class DisiWrapper {
  public final DocIdSetIterator iterator;
  // Same as `iterator` if `iterator` is an instance of PostingsEnum, null otherwise
  final PostingsEnum postingsEnum;
  public final Scorer scorer;
  public final Scorable scorable;
  public final long cost;
  public final float matchCost; // the match cost for two-phase iterators, 0 otherwise
  public int doc; // the current doc, used for comparison
  public DisiWrapper next; // reference to a next element, see #topList

  // An approximation of the iterator, or the iterator itself if it does not
  // support two-phase iteration
  public final DocIdSetIterator approximation;
  // A two-phase view of the iterator, or null if the iterator does not support
  // two-phase iteration
  public final TwoPhaseIterator twoPhaseView;

  // For WANDScorer
  long scaledMaxScore;

  // for MaxScoreBulkScorer
  float maxWindowScore;

  // for CombinedFieldQuery (BM25F)
  final float weight;

  public DisiWrapper(Scorer scorer, boolean impacts) {
    this(scorer, impacts, 1f);
  }

  DisiWrapper(Scorer scorer, boolean impacts, float weight) {
    this.scorer = Objects.requireNonNull(scorer);
    this.scorable = ScorerUtil.likelyTermScorer(scorer);
    if (impacts) {
      this.iterator = ScorerUtil.likelyImpactsEnum(scorer.iterator());
    } else {
      this.iterator = scorer.iterator();
    }
    this.postingsEnum = iterator instanceof PostingsEnum pe ? pe : null;
    this.cost = iterator.cost();
    this.doc = -1;
    this.twoPhaseView = scorer.twoPhaseIterator();

    if (twoPhaseView != null) {
      approximation = twoPhaseView.approximation();
      matchCost = twoPhaseView.matchCost();
    } else {
      approximation = iterator;
      matchCost = 0f;
    }

    this.weight = weight;
  }
}
