

package org.tensorflow.envisager;

import org.tensorflow.envisager.Classifier.Recognition;

import java.util.List;

public interface ResultsView {
  public void setResults(final List<Recognition> results);
}
