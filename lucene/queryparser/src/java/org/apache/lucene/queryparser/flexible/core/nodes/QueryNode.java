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
package org.apache.lucene.queryparser.flexible.core.nodes;

import java.util.List;
import java.util.Map;
import org.apache.lucene.queryparser.flexible.core.parser.EscapeQuerySyntax;

/** A {@link QueryNode} is a interface implemented by all nodes on a QueryNode tree. */
public interface QueryNode {

  /** convert to a query string understood by the query parser */
  CharSequence toQueryString(EscapeQuerySyntax escapeSyntaxParser);

  /** for printing */
  @Override
  String toString();

  /** get Children nodes */
  List<QueryNode> getChildren();

  /** verify if a node is a Leaf node */
  boolean isLeaf();

  /** verify if a node contains a tag */
  boolean containsTag(String tagName);

  /** Returns object stored under that tag name */
  Object getTag(String tagName);

  QueryNode getParent();

  /**
   * Recursive clone the QueryNode tree. The tags are not copied to the new tree when you call the
   * cloneTree() method.
   *
   * @return the cloned tree
   */
  QueryNode cloneTree() throws CloneNotSupportedException;

  // Below are the methods that can change state of a QueryNode
  // Write Operations (not Thread Safe)

  /** add a new child to a non Leaf node */
  void add(QueryNode child);

  void add(List<QueryNode> children);

  /** reset the children of a node */
  void set(List<QueryNode> children);

  /**
   * Associate the specified value with the specified tagName. If the tagName already exists, the
   * old value is replaced. The tagName and value cannot be null. tagName will be converted to
   * lowercase.
   */
  void setTag(String tagName, Object value);

  /** Unset a tag. tagName will be converted to lowercase. */
  void unsetTag(String tagName);

  /**
   * @return a map containing all tags attached to this query node
   */
  Map<String, Object> getTagMap();

  /** Removes this query node from its parent. */
  void removeFromParent();

  /**
   * Remove a child node
   *
   * @param childNode Which child to remove
   */
  void removeChildren(QueryNode childNode);
}
