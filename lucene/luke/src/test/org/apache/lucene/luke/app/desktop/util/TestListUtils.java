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

package org.apache.lucene.luke.app.desktop.util;

import java.util.Arrays;
import java.util.List;
import javax.swing.JList;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestListUtils extends LuceneTestCase {

  @Test
  public void testGetAllItems() {
    JList<String> list = new JList<>(new String[] {"Item 1", "Item 2"});
    List<String> items = ListUtils.getAllItems(list);
    assertEquals(Arrays.asList("Item 1", "Item 2"), items);
    // test mutability
    items.add("Item 3");
    assertEquals(Arrays.asList("Item 1", "Item 2", "Item 3"), items);
  }
}
