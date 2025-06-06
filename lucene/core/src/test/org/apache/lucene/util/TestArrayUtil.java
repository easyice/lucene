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
package org.apache.lucene.util;

import static org.apache.lucene.util.ArrayUtil.copyOfSubArray;
import static org.apache.lucene.util.ArrayUtil.growExact;
import static org.apache.lucene.util.ArrayUtil.growInRange;
import static org.apache.lucene.util.ArrayUtil.oversize;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.hamcrest.Matchers.sameInstance;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;

public class TestArrayUtil extends LuceneTestCase {

  // Ensure ArrayUtil.getNextSize gives linear amortized cost of realloc/copy
  public void testGrowth() {
    int currentSize = 0;
    long copyCost = 0;

    // Make sure ArrayUtil hits Integer.MAX_VALUE, if we insist:
    while (currentSize != ArrayUtil.MAX_ARRAY_LENGTH) {
      int nextSize = ArrayUtil.oversize(1 + currentSize, RamUsageEstimator.NUM_BYTES_OBJECT_REF);
      assertThat(nextSize, greaterThan(currentSize));
      if (currentSize > 0) {
        copyCost += currentSize;
        double copyCostPerElement = ((double) copyCost) / currentSize;
        assertThat(copyCostPerElement, lessThan(10.0));
      }
      currentSize = nextSize;
    }
  }

  public void testMaxSize() {
    // intentionally pass invalid elemSizes:
    for (int elemSize = 0; elemSize < 10; elemSize++) {
      assertEquals(
          ArrayUtil.MAX_ARRAY_LENGTH, ArrayUtil.oversize(ArrayUtil.MAX_ARRAY_LENGTH, elemSize));
      assertEquals(
          ArrayUtil.MAX_ARRAY_LENGTH, ArrayUtil.oversize(ArrayUtil.MAX_ARRAY_LENGTH - 1, elemSize));
    }
  }

  public void testTooBig() {
    expectThrows(
        IllegalArgumentException.class,
        () -> {
          ArrayUtil.oversize(ArrayUtil.MAX_ARRAY_LENGTH + 1, 1);
        });
  }

  public void testExactLimit() {
    assertEquals(ArrayUtil.MAX_ARRAY_LENGTH, ArrayUtil.oversize(ArrayUtil.MAX_ARRAY_LENGTH, 1));
  }

  public void testInvalidElementSizes() {
    final Random rnd = random();
    final int num = atLeast(10000);
    for (int iter = 0; iter < num; iter++) {
      final int minTargetSize = rnd.nextInt(ArrayUtil.MAX_ARRAY_LENGTH);
      final int elemSize = rnd.nextInt(11);
      final int v = ArrayUtil.oversize(minTargetSize, elemSize);
      assertThat(v, greaterThanOrEqualTo(minTargetSize));
    }
  }

  private static int parseInt(String s) {
    int start = random().nextInt(5);
    char[] chars = new char[s.length() + start + random().nextInt(4)];
    s.getChars(0, s.length(), chars, start);
    return ArrayUtil.parseInt(chars, start, s.length());
  }

  public void testParseInt() throws Exception {
    expectThrows(
        NumberFormatException.class,
        () -> {
          parseInt("");
        });

    expectThrows(
        NumberFormatException.class,
        () -> {
          parseInt("foo");
        });

    expectThrows(
        NumberFormatException.class,
        () -> {
          parseInt(String.valueOf(Long.MAX_VALUE));
        });

    expectThrows(
        NumberFormatException.class,
        () -> {
          parseInt("0.34");
        });

    assertThat(parseInt("1"), equalTo(1));
    assertThat(parseInt("-10000"), equalTo(-10000));
    assertThat(parseInt("1923"), equalTo(1923));
    assertThat(parseInt("-1"), equalTo(-1));
    assertThat(ArrayUtil.parseInt("foo 1923 bar".toCharArray(), 4, 4), equalTo(1923));
  }

  private Integer[] createRandomArray(int maxSize) {
    final Random rnd = random();
    final Integer[] a = new Integer[rnd.nextInt(maxSize) + 1];
    for (int i = 0; i < a.length; i++) {
      a[i] = Integer.valueOf(rnd.nextInt(a.length));
    }
    return a;
  }

  public void testIntroSort() {
    int num = atLeast(50);
    for (int i = 0; i < num; i++) {
      Integer[] a1 = createRandomArray(2000), a2 = a1.clone();
      ArrayUtil.introSort(a1);
      Arrays.sort(a2);
      assertArrayEquals(a2, a1);

      a1 = createRandomArray(2000);
      a2 = a1.clone();
      ArrayUtil.introSort(a1, Collections.reverseOrder());
      Arrays.sort(a2, Collections.reverseOrder());
      assertArrayEquals(a2, a1);
      // reverse back, so we can test that completely backwards sorted array (worst case) is
      // working:
      ArrayUtil.introSort(a1);
      Arrays.sort(a2);
      assertArrayEquals(a2, a1);
    }
  }

  private Integer[] createSparseRandomArray(int maxSize) {
    final Random rnd = random();
    final Integer[] a = new Integer[rnd.nextInt(maxSize) + 1];
    for (int i = 0; i < a.length; i++) {
      a[i] = Integer.valueOf(rnd.nextInt(2));
    }
    return a;
  }

  // This is a test for LUCENE-3054 (which fails without the merge sort fall back with stack
  // overflow in most cases)
  public void testQuickToHeapSortFallback() {
    int num = atLeast(10);
    for (int i = 0; i < num; i++) {
      Integer[] a1 = createSparseRandomArray(40000), a2 = a1.clone();
      ArrayUtil.introSort(a1);
      Arrays.sort(a2);
      assertArrayEquals(a2, a1);
    }
  }

  public void testTimSort() {
    int num = atLeast(50);
    for (int i = 0; i < num; i++) {
      Integer[] a1 = createRandomArray(2000), a2 = a1.clone();
      ArrayUtil.timSort(a1);
      Arrays.sort(a2);
      assertArrayEquals(a2, a1);

      a1 = createRandomArray(2000);
      a2 = a1.clone();
      ArrayUtil.timSort(a1, Collections.reverseOrder());
      Arrays.sort(a2, Collections.reverseOrder());
      assertArrayEquals(a2, a1);
      // reverse back, so we can test that completely backwards sorted array (worst case) is
      // working:
      ArrayUtil.timSort(a1);
      Arrays.sort(a2);
      assertArrayEquals(a2, a1);
    }
  }

  record Item(int val, int order) implements Comparable<Item> {

    @Override
    public int compareTo(Item other) {
      return this.order - other.order;
    }

    @Override
    public String toString() {
      return Integer.toString(val);
    }
  }

  public void testMergeSortStability() {
    final Random rnd = random();
    Item[] items = new Item[100];
    for (int i = 0; i < items.length; i++) {
      // half of the items have value but same order. The value of this items is sorted,
      // so they should always be in order after sorting.
      // The other half has defined order, but no (-1) value (they should appear after
      // all above, when sorted).
      final boolean equal = rnd.nextBoolean();
      items[i] = new Item(equal ? (i + 1) : -1, equal ? 0 : (rnd.nextInt(1000) + 1));
    }

    if (VERBOSE) System.out.println("Before: " + Arrays.toString(items));
    // if you replace this with ArrayUtil.quickSort(), test should fail:
    ArrayUtil.timSort(items);
    if (VERBOSE) System.out.println("Sorted: " + Arrays.toString(items));

    Item last = items[0];
    for (int i = 1; i < items.length; i++) {
      final Item act = items[i];
      if (act.order == 0) {
        // order of "equal" items should be not mixed up
        assertThat(act.val, greaterThan(last.val));
      }
      assertThat(act.order, greaterThanOrEqualTo(last.order));
      last = act;
    }
  }

  public void testTimSortStability() {
    final Random rnd = random();
    Item[] items = new Item[100];
    for (int i = 0; i < items.length; i++) {
      // half of the items have value but same order. The value of this items is sorted,
      // so they should always be in order after sorting.
      // The other half has defined order, but no (-1) value (they should appear after
      // all above, when sorted).
      final boolean equal = rnd.nextBoolean();
      items[i] = new Item(equal ? (i + 1) : -1, equal ? 0 : (rnd.nextInt(1000) + 1));
    }

    if (VERBOSE) System.out.println("Before: " + Arrays.toString(items));
    // if you replace this with ArrayUtil.quickSort(), test should fail:
    ArrayUtil.timSort(items);
    if (VERBOSE) System.out.println("Sorted: " + Arrays.toString(items));

    Item last = items[0];
    for (int i = 1; i < items.length; i++) {
      final Item act = items[i];
      if (act.order == 0) {
        // order of "equal" items should be not mixed up
        assertThat(act.val, greaterThan(last.val));
      }
      assertThat(act.order, greaterThanOrEqualTo(last.order));
      last = act;
    }
  }

  // should produce no exceptions
  public void testEmptyArraySort() {
    Integer[] a = new Integer[0];
    ArrayUtil.introSort(a);
    ArrayUtil.timSort(a);
    ArrayUtil.introSort(a, Collections.reverseOrder());
    ArrayUtil.timSort(a, Collections.reverseOrder());
  }

  public void testSelect() {
    for (int iter = 0; iter < 100; ++iter) {
      doTestSelect();
    }
  }

  private void doTestSelect() {
    final int from = random().nextInt(5);
    final int to = from + TestUtil.nextInt(random(), 1, 10000);
    final int max = random().nextBoolean() ? random().nextInt(100) : random().nextInt(100000);
    Integer[] arr = new Integer[from + to + random().nextInt(5)];
    for (int i = 0; i < arr.length; ++i) {
      arr[i] = TestUtil.nextInt(random(), 0, max);
    }
    final int k = TestUtil.nextInt(random(), from, to - 1);

    Integer[] expected = arr.clone();
    Arrays.sort(expected, from, to);

    Integer[] actual = arr.clone();
    ArrayUtil.select(actual, from, to, k, Comparator.naturalOrder());

    assertEquals(expected[k], actual[k]);
    for (int i = 0; i < actual.length; ++i) {
      if (i < from || i >= to) {
        assertThat(actual[i], sameInstance(arr[i]));
      } else if (i <= k) {
        assertThat(actual[i], lessThanOrEqualTo(actual[k]));
      } else {
        assertThat(actual[i], greaterThanOrEqualTo(actual[k]));
      }
    }
  }

  public void testGrowExact() {
    assertArrayEquals(new short[] {1, 2, 3, 0}, growExact(new short[] {1, 2, 3}, 4));
    assertArrayEquals(new short[] {1, 2, 3, 0, 0}, growExact(new short[] {1, 2, 3}, 5));
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> growExact(new short[] {1, 2, 3}, random().nextInt(3)));

    assertArrayEquals(new int[] {1, 2, 3, 0}, growExact(new int[] {1, 2, 3}, 4));
    assertArrayEquals(new int[] {1, 2, 3, 0, 0}, growExact(new int[] {1, 2, 3}, 5));
    expectThrows(
        IndexOutOfBoundsException.class, () -> growExact(new int[] {1, 2, 3}, random().nextInt(3)));

    assertArrayEquals(new long[] {1, 2, 3, 0}, growExact(new long[] {1, 2, 3}, 4));
    assertArrayEquals(new long[] {1, 2, 3, 0, 0}, growExact(new long[] {1, 2, 3}, 5));
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> growExact(new long[] {1, 2, 3}, random().nextInt(3)));

    assertArrayEquals(
        new float[] {0.1f, 0.2f, 0.3f, 0}, growExact(new float[] {0.1f, 0.2f, 0.3f}, 4), 0.001f);
    assertArrayEquals(
        new float[] {0.1f, 0.2f, 0.3f, 0, 0}, growExact(new float[] {0.1f, 0.2f, 0.3f}, 5), 0.001f);
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> growExact(new float[] {1, 2, 3}, random().nextInt(3)));

    assertArrayEquals(
        new double[] {0.1, 0.2, 0.3, 0.0}, growExact(new double[] {0.1, 0.2, 0.3}, 4), 0.001);
    assertArrayEquals(
        new double[] {0.1, 0.2, 0.3, 0.0, 0.0}, growExact(new double[] {0.1, 0.2, 0.3}, 5), 0.001);
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> growExact(new double[] {0.1, 0.2, 0.3}, random().nextInt(3)));

    assertArrayEquals(new byte[] {1, 2, 3, 0}, growExact(new byte[] {1, 2, 3}, 4));
    assertArrayEquals(new byte[] {1, 2, 3, 0, 0}, growExact(new byte[] {1, 2, 3}, 5));
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> growExact(new byte[] {1, 2, 3}, random().nextInt(3)));

    assertArrayEquals(new char[] {'a', 'b', 'c', '\0'}, growExact(new char[] {'a', 'b', 'c'}, 4));
    assertArrayEquals(
        new char[] {'a', 'b', 'c', '\0', '\0'}, growExact(new char[] {'a', 'b', 'c'}, 5));
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> growExact(new byte[] {'a', 'b', 'c'}, random().nextInt(3)));

    assertArrayEquals(
        new String[] {"a1", "b2", "c3", null}, growExact(new String[] {"a1", "b2", "c3"}, 4));
    assertArrayEquals(
        new String[] {"a1", "b2", "c3", null, null}, growExact(new String[] {"a1", "b2", "c3"}, 5));
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> growExact(new String[] {"a", "b", "c"}, random().nextInt(3)));
  }

  public void testGrowInRange() {
    int[] array = new int[] {1, 2, 3};

    // If minLength is negative, maxLength does not matter
    expectThrows(AssertionError.class, () -> growInRange(array, -1, 4));
    expectThrows(AssertionError.class, () -> growInRange(array, -1, 0));
    expectThrows(AssertionError.class, () -> growInRange(array, -1, -1));

    // If minLength > maxLength, we throw an exception
    expectThrows(IllegalArgumentException.class, () -> growInRange(array, 1, 0));
    expectThrows(IllegalArgumentException.class, () -> growInRange(array, 4, 3));
    expectThrows(IllegalArgumentException.class, () -> growInRange(array, 5, 4));

    // If minLength is sufficient, we return the array
    assertSame(array, growInRange(array, 1, 4));
    assertSame(array, growInRange(array, 1, 2));
    assertSame(array, growInRange(array, 1, 1));

    int minLength = 4;
    int maxLength = Integer.MAX_VALUE;

    // The array grows normally if maxLength permits
    assertEquals(
        oversize(minLength, Integer.BYTES),
        growInRange(new int[] {1, 2, 3}, minLength, maxLength).length);

    // The array grows to maxLength if maxLength is limiting
    assertEquals(minLength, growInRange(new int[] {1, 2, 3}, minLength, minLength).length);
  }

  public void testGrowInRangeFloat() {
    float[] array = new float[] {1f, 2f, 3f};

    // If minLength is negative, maxLength does not matter
    expectThrows(AssertionError.class, () -> growInRange(array, -1, 4));
    expectThrows(AssertionError.class, () -> growInRange(array, -1, 0));
    expectThrows(AssertionError.class, () -> growInRange(array, -1, -1));

    // If minLength > maxLength, we throw an exception
    expectThrows(IllegalArgumentException.class, () -> growInRange(array, 1, 0));
    expectThrows(IllegalArgumentException.class, () -> growInRange(array, 4, 3));
    expectThrows(IllegalArgumentException.class, () -> growInRange(array, 5, 4));

    // If minLength is sufficient, we return the array
    assertSame(array, growInRange(array, 1, 4));
    assertSame(array, growInRange(array, 1, 2));
    assertSame(array, growInRange(array, 1, 1));

    int minLength = 4;
    int maxLength = Integer.MAX_VALUE;

    // The array grows normally if maxLength permits
    assertEquals(
        oversize(minLength, Float.BYTES),
        growInRange(new float[] {1f, 2f, 3f}, minLength, maxLength).length);

    // The array grows to maxLength if maxLength is limiting
    assertEquals(minLength, growInRange(new float[] {1f, 2f, 3f}, minLength, minLength).length);
  }

  public void testCopyOfSubArray() {
    short[] shortArray = {1, 2, 3};
    assertArrayEquals(new short[] {1}, copyOfSubArray(shortArray, 0, 1));
    assertArrayEquals(new short[] {1, 2, 3}, copyOfSubArray(shortArray, 0, 3));
    assertEquals(0, copyOfSubArray(shortArray, 0, 0).length);
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> copyOfSubArray(shortArray, 0, 4 + random().nextInt(10)));

    int[] intArray = {1, 2, 3};
    assertArrayEquals(new int[] {1, 2}, copyOfSubArray(intArray, 0, 2));
    assertArrayEquals(new int[] {1, 2, 3}, copyOfSubArray(intArray, 0, 3));
    assertEquals(0, copyOfSubArray(intArray, 1, 1).length);
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> copyOfSubArray(intArray, 1, 4 + random().nextInt(10)));

    long[] longArray = {1, 2, 3};
    assertArrayEquals(new long[] {2}, copyOfSubArray(longArray, 1, 2));
    assertArrayEquals(new long[] {1, 2, 3}, copyOfSubArray(longArray, 0, 3));
    assertEquals(0, copyOfSubArray(longArray, 2, 2).length);
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> copyOfSubArray(longArray, 2, 4 + random().nextInt(10)));

    float[] floatArray = {0.1f, 0.2f, 0.3f};
    assertArrayEquals(new float[] {0.2f, 0.3f}, copyOfSubArray(floatArray, 1, 3), 0.001f);
    assertArrayEquals(new float[] {0.1f, 0.2f, 0.3f}, copyOfSubArray(floatArray, 0, 3), 0.001f);
    assertEquals(0, copyOfSubArray(floatArray, 0, 0).length);
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> copyOfSubArray(floatArray, 3, 4 + random().nextInt(10)));

    double[] doubleArray = {0.1, 0.2, 0.3};
    assertArrayEquals(new double[] {0.3}, copyOfSubArray(doubleArray, 2, 3), 0.001);
    assertArrayEquals(new double[] {0.1, 0.2, 0.3}, copyOfSubArray(doubleArray, 0, 3), 0.001);
    assertEquals(0, copyOfSubArray(doubleArray, 1, 1).length);
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> copyOfSubArray(doubleArray, 0, 4 + random().nextInt(10)));

    byte[] byteArray = {1, 2, 3};
    assertArrayEquals(new byte[] {1}, copyOfSubArray(byteArray, 0, 1));
    assertArrayEquals(new byte[] {1, 2, 3}, copyOfSubArray(byteArray, 0, 3));
    assertEquals(0, copyOfSubArray(byteArray, 1, 1).length);
    expectThrows(
        IndexOutOfBoundsException.class,
        () -> copyOfSubArray(byteArray, 1, 4 + random().nextInt(10)));

    char[] charArray = {'a', 'b', 'c'};
    assertArrayEquals(new char[] {'a', 'b'}, copyOfSubArray(charArray, 0, 2));
    assertArrayEquals(new char[] {'a', 'b', 'c'}, copyOfSubArray(charArray, 0, 3));
    assertEquals(0, copyOfSubArray(charArray, 1, 1).length);
    expectThrows(IndexOutOfBoundsException.class, () -> copyOfSubArray(charArray, 3, 4));

    String[] objectArray = {"a1", "b2", "c3"};
    assertArrayEquals(new String[] {"a1"}, copyOfSubArray(objectArray, 0, 1));
    assertArrayEquals(new String[] {"a1", "b2", "c3"}, copyOfSubArray(objectArray, 0, 3));
    assertEquals(0, copyOfSubArray(objectArray, 1, 1).length);
    expectThrows(IndexOutOfBoundsException.class, () -> copyOfSubArray(objectArray, 2, 5));
  }

  public void testCompareUnsigned4() {
    int aOffset = TestUtil.nextInt(random(), 0, 3);
    byte[] a = new byte[Integer.BYTES + aOffset];
    int bOffset = TestUtil.nextInt(random(), 0, 3);
    byte[] b = new byte[Integer.BYTES + bOffset];

    for (int i = 0; i < Integer.BYTES; ++i) {
      a[aOffset + i] = (byte) random().nextInt(1 << 8);
      do {
        b[bOffset + i] = (byte) random().nextInt(1 << 8);
      } while (b[bOffset + i] == a[aOffset + i]);
    }

    for (int i = 0; i < Integer.BYTES; ++i) {
      int expected =
          Arrays.compareUnsigned(
              a, aOffset, aOffset + Integer.BYTES, b, bOffset, bOffset + Integer.BYTES);
      int actual = ArrayUtil.compareUnsigned4(a, aOffset, b, bOffset);
      assertEquals(Integer.signum(expected), Integer.signum(actual));
      b[bOffset + i] = a[aOffset + i];
    }

    assertEquals(0, ArrayUtil.compareUnsigned4(a, aOffset, b, bOffset));
  }

  public void testCompareUnsigned8() {
    int aOffset = TestUtil.nextInt(random(), 0, 7);
    byte[] a = new byte[Long.BYTES + aOffset];
    int bOffset = TestUtil.nextInt(random(), 0, 7);
    byte[] b = new byte[Long.BYTES + bOffset];

    for (int i = 0; i < Long.BYTES; ++i) {
      a[aOffset + i] = (byte) random().nextInt(1 << 8);
      do {
        b[bOffset + i] = (byte) random().nextInt(1 << 8);
      } while (b[bOffset + i] == a[aOffset + i]);
    }

    for (int i = 0; i < Long.BYTES; ++i) {
      int expected =
          Arrays.compareUnsigned(
              a, aOffset, aOffset + Long.BYTES, b, bOffset, bOffset + Long.BYTES);
      int actual = ArrayUtil.compareUnsigned8(a, aOffset, b, bOffset);
      assertEquals(Integer.signum(expected), Integer.signum(actual));
      b[bOffset + i] = a[aOffset + i];
    }

    assertEquals(0, ArrayUtil.compareUnsigned8(a, aOffset, b, bOffset));
  }
}
