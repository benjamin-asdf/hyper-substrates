(ns ftlm.disco.middlewares.time-warp
  (:require [quil.core :as q :include-macros true]))

(defn timewarp-state []
  {:history [] :last-tick nil})

(defn add-vec
  "Add two or more vectors together"
  [& args]
  (when (seq args) (apply mapv + args)))

(defn mult-vec [& args]
  (when (seq args)
    (apply mapv * args)))

(defn update-time-warp
  [{:keys
    [last-tick history] :as time-warp}]
  (if (or
       (not last-tick)
       (< 5 (- (q/seconds) last-tick))
       ))

  )

(defn time-warp-update
  [state time-warp]
  (if (:warps? time-warp)
    (or (last (:history time-warp)) state)
    state))

(defn update-state
  [user-update state]
  (let [time-warp (:time-warp state)]
    (-> (user-update state)
        (time-warp-update time-warp)
        (assoc :time-warp (update-time-warp time-warp)))))

(defn setup-picture-jitter
  [user-setup user-settings]
  (let [initial-state (merge (timewarp-state)
                             user-settings)]
    (update-in
     (user-setup)
     [:time-warp]
     #(merge initial-state %))))

(defn time-warp
  [options]
  (let [user-settings (:time-warp options)
        user-update (:update options (fn [state]))
        user-setup (:setup options (fn [] {}))]
    (def user-setup user-setup)
    options
    (assoc options
      :setup (partial setup-picture-jitter
                      user-setup
                      user-settings)
      :update (partial update-state user-update))))
