(ns ftlm.vehicles.art.grid
  (:require
   [ftlm.vehicles.art.lib :as lib :refer [*dt*]]
   [quil.core :as q :include-macros true]
   [quil.middleware :as m]
   [ftlm.vehicles.art.extended :as elib]
   [ftlm.vehicles.art.controls :as controls :refer
    [versions]]
   [ftlm.vehicles.art.user-controls :as
    user-controls]))

(defmethod lib/draw-entity :grid
  [{:keys [transform no-stroke? stroke draw-element elements
           spacing grid-width]}]
  (let [[x y] (:pos transform)]
    (doall
     (for [i (range (count elements))
           :let [coll (mod i grid-width)
                 row (quot i grid-width)]]
       (let [x (+ x (* coll spacing))
             y (+ y (* row spacing))]
         (q/with-translation
           [x y]
           (draw-element (elements i))))))))
