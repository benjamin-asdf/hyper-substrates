(ns ftlm.vehicles.art.util)

(defn descending [a b] (compare b a))
(def ascending compare)

(defmacro by*
  ([a b key ordering & keys-orderings]
   `(let [order# (~ordering (~key ~a) (~key ~b))]
      (if (zero? order#)
        (by* ~a ~b ~@keys-orderings)
        order#)))
  ([a b key ordering]
   `(~ordering (~key ~a) (~key ~b))))

(defmacro by [& keys-orderings]
  `(fn [a# b#]
     (by* a# b# ~@keys-orderings)))
