(ns animismic.lib.vehicles
  (:require
   [fastmath.core :as fm]
   [ftlm.vehicles.art.lib :refer [*dt*] :as lib]))

(defn vehicle-temperature
  [state]
  (transduce
   (comp (filter :body?) (map :velocity) (map fm/abs))
   +
   0
   (lib/entities @lib/the-state)))
