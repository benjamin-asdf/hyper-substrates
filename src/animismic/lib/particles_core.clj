(ns animismic.lib.particles-core
  (:require [animismic.lib.particles :as p]))

(def particle-update
  [:particle
   (fn [e s _]
     (update e :particle-field p/update-grid-field))])
